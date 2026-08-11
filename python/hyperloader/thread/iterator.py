"""Strict-order delivery from declared thread-safe dataset calls."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future
from queue import Empty, SimpleQueue
from typing import Any

from ..process.sizing import delivery_length, frontier_depth
from ..state import (
    DeliveredBatchState,
    decode_delivered_bitmap,
    resume_sample_position,
)
from ..telemetry.delivery import build_delivery_telemetry
from .pool import ThreadPool


class ThreadIterator(Iterator[Any]):
    """Run bounded dataset calls in threads and commit in sampler order."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self._epoch = loader._epoch
        self._length = delivery_length(loader)
        resume_position = resume_sample_position(loader, self._length)
        self._on_completion = loader.delivery == "on-completion"
        width = loader.batch_size or 1
        total_batches = (self._length + width - 1) // width
        self._restored_batches = (
            decode_delivered_bitmap(
                loader._resume_cursor_batches,
                loader._resume_delivered_bitmap,
                total_batches,
            )
            if self._on_completion
            else set()
        )
        restored_samples = sum(
            min(width, self._length - ordinal * width)
            for ordinal in self._restored_batches
        )
        self._position = resume_position + restored_samples
        self._next_submit = resume_position
        restored_span = (
            max(self._restored_batches) * width + width - resume_position
            if self._restored_batches
            else 0
        )
        self._depth = min(
            self._length - resume_position,
            max(frontier_depth(loader), restored_span),
        )
        self._complete = False
        self._valid = True
        self._futures: dict[int, Future[tuple[Any, int]]] = {}
        self._completed: SimpleQueue[int] = SimpleQueue()
        self._ready_batches: deque[int] = deque()
        self._queued_batches: set[int] = set()
        self._delivered = DeliveredBatchState(
            loader._resume_cursor_batches, self._restored_batches
        )
        self._delivery_telemetry = build_delivery_telemetry(loader)
        if self._length:
            if loader._thread_pool is None:
                loader._thread_pool = ThreadPool(
                    loader._execution_dataset,
                    loader.num_workers,
                    loader.root_seed,
                    loader.worker_init_fn,
                    loader._telemetry,
                )
            self._fill_frontier()

    def __iter__(self) -> ThreadIterator:
        return self

    def __next__(self) -> Any:
        started = time.perf_counter_ns()
        previous_position = self._position
        try:
            value = self._next_value()
        except StopIteration:
            raise
        except BaseException:
            self._loader.close()
            raise
        if self._delivery_telemetry is not None:
            samples = self._position - previous_position
            self._delivery_telemetry.record_delivery(samples, 0, started)
        return value

    def _next_value(self) -> Any:
        if not self._valid:
            raise RuntimeError("thread iterator is no longer active")
        if self._position >= self._length:
            self._finish_epoch()
            raise StopIteration
        if self._on_completion:
            ordinal, value, delivered = self._next_ready_batch()
            self._position += delivered
            self._delivered.mark(ordinal)
            self._loader._epoch_state.mark_delivered(self._epoch)
            return value
        batch_size = self._loader.batch_size
        if batch_size is None:
            value = self._next_sample()
            self._loader._epoch_state.mark_delivered(self._epoch)
            return value
        stop = min(self._position + batch_size, self._length)
        values = [self._next_sample() for _ in range(self._position, stop)]
        self._loader._epoch_state.mark_delivered(self._epoch)
        return self._loader._collate_batch(values)

    def _next_sample(self) -> Any:
        position = self._position
        future = self._futures.pop(position)
        timeout = None if self._loader.timeout == 0 else self._loader.timeout
        value, cost_ns = future.result(timeout=timeout)
        profile = self._loader._cost_profile
        if profile is not None:
            profile.observe(position, cost_ns)
        self._position += 1
        self._fill_frontier()
        return value

    def _next_ready_batch(self) -> tuple[int, Any, int]:
        batch_size = self._loader.batch_size
        width = batch_size or 1
        timeout = None if self._loader.timeout == 0 else self._loader.timeout
        while not self._ready_batches:
            try:
                position = self._completed.get(timeout=timeout)
            except Empty as error:
                raise RuntimeError(
                    f"DataLoader timed out after {self._loader.timeout} seconds"
                ) from error
            ordinal = position // width
            if ordinal in self._queued_batches:
                continue
            start = ordinal * width
            stop = min(start + width, self._length)
            if all(
                sample in self._futures and self._futures[sample].done()
                for sample in range(start, stop)
            ):
                self._queued_batches.add(ordinal)
                self._ready_batches.append(ordinal)

        ordinal = self._ready_batches.popleft()
        self._queued_batches.remove(ordinal)
        start = ordinal * width
        stop = min(start + width, self._length)
        values = []
        profile = self._loader._cost_profile
        for position in range(start, stop):
            value, cost_ns = self._futures.pop(position).result()
            if profile is not None:
                profile.observe(position, cost_ns)
            values.append(value)
        self._fill_frontier()
        value = values[0] if batch_size is None else self._loader._collate_batch(values)
        return ordinal, value, stop - start

    def _fill_frontier(self) -> None:
        pool = self._loader._thread_pool
        while self._next_submit < self._length and len(self._futures) < self._depth:
            position = self._next_submit
            width = self._loader.batch_size or 1
            if position // width in self._restored_batches:
                self._next_submit += 1
                continue
            coordinate = self._loader._map_coordinate(position)
            index = self._loader._map_index(self._epoch, position)
            self._futures[position] = pool.submit(
                self._epoch, position, index, coordinate
            )
            if self._on_completion:
                self._futures[position].add_done_callback(
                    lambda _future, completed=position: self._completed.put(completed)
                )
            self._next_submit += 1

    def _finish_epoch(self) -> None:
        if self._complete:
            return
        self._loader._epoch_state.complete(self._epoch)
        from hyperloader.profile import save_cost_profile

        save_cost_profile(self._loader)
        if self._delivery_telemetry is not None:
            self._delivery_telemetry.finish_epoch(self._epoch)
        self._complete = True

    def _flush_telemetry(self) -> None:
        if self._delivery_telemetry is not None:
            self._delivery_telemetry.flush()

    @property
    def complete(self) -> bool:
        """Report whether exhaustion advanced the loader epoch."""
        return self._complete

    @property
    def coordinate_epoch(self) -> int:
        """Return the epoch carried by this iterator's checkpoint coordinate."""
        return self._epoch

    @property
    def delivered_batches(self) -> int:
        """Return the strict delivered-batch prefix count."""
        if self._on_completion:
            return self._delivered.base
        batch_size = self._loader.batch_size or 1
        return (self._position + batch_size - 1) // batch_size

    @property
    def delivered_bitmap(self) -> bytes:
        """Encode delivered batches beyond the contiguous prefix."""
        return self._delivered.bitmap() if self._on_completion else b""

    def invalidate(self) -> None:
        """Prevent a replaced iterator from committing more results."""
        self._valid = False

"""Strict-order delivery from declared thread-safe dataset calls."""

from __future__ import annotations

import time
from collections.abc import Iterator
from concurrent.futures import Future
from typing import Any

from ..process.sizing import delivery_length, frontier_depth
from ..telemetry.delivery import build_delivery_telemetry
from .pool import ThreadPool


class ThreadIterator(Iterator[Any]):
    """Run bounded dataset calls in threads and commit in sampler order."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self._epoch = loader._epoch
        self._position = 0
        self._next_submit = 0
        self._length = delivery_length(loader)
        self._depth = min(self._length, frontier_depth(loader))
        self._complete = False
        self._valid = True
        self._futures: dict[int, Future[tuple[Any, int]]] = {}
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

    def _fill_frontier(self) -> None:
        pool = self._loader._thread_pool
        while self._next_submit < self._length and len(self._futures) < self._depth:
            position = self._next_submit
            index = self._loader._plan.index(
                self._loader.root_seed, self._epoch, position
            )
            self._futures[position] = pool.submit(self._epoch, position, index)
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

    def invalidate(self) -> None:
        """Prevent a replaced iterator from committing more results."""
        self._valid = False

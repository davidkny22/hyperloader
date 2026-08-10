"""Strict-order delivery over a bounded native execution frontier."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from hyperloader import _hyperloader

from .batching import unwrap_batch_payload
from .exceptions import WorkerDied
from .factory import prepare_process_pool
from .pool import POLL_SECONDS
from .sizing import delivery_length, frontier_depth


class ProcessIterator(Iterator[Any]):
    """Execute ahead within a fixed frontier and commit in sampler order."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self._epoch = loader._epoch
        self._position = 0
        self._length = delivery_length(loader)
        self._complete = False
        self._valid = True
        self._ready: dict[int, tuple[int, bytes, int]] = {}
        self._schedule: Any = None
        if self._length:
            depth = frontier_depth(loader)
            prepare_process_pool(loader)
            self._schedule = _hyperloader._StaticSchedule(
                0,
                self._length,
                depth,
                loader._process_pool.worker_count,
                loader._process_pool.batch_size or 1,
            )
            self._fill_frontier()

    def __iter__(self) -> ProcessIterator:
        return self

    def __next__(self) -> Any:
        if not self._valid:
            raise RuntimeError("process iterator is no longer active")
        try:
            if self._position >= self._length:
                self._finish_epoch()
                raise StopIteration
            batch_size = self._loader.batch_size
            if batch_size is None:
                position = self._position
                self._position += 1
                return self._next_sample(position)
            stop = min(self._position + batch_size, self._length)
            batch = self._next_batch(self._position, stop)
            self._position = stop
            return batch
        except StopIteration:
            raise
        except WorkerDied:
            raise
        except BaseException:
            self._loader.close()
            raise

    def _next_sample(self, expected_position: int) -> Any:
        status, payload, worker = self._next_completion(expected_position)
        return self._loader._process_pool.decode(status, payload, worker)

    def _next_batch(self, start: int, stop: int) -> Any:
        pool = self._loader._process_pool
        samples = []
        saw_acknowledgement = False
        materialized = None
        saw_materialized = False
        for position in range(start, stop):
            status, payload, worker = self._next_completion(position)
            if status != 0:
                pool.decode(status, payload, worker)
            if not payload:
                saw_acknowledgement = True
                continue
            batch_payload = unwrap_batch_payload(payload)
            if batch_payload is not None:
                if position != stop - 1:
                    raise RuntimeError(
                        "worker batch payload arrived before its final position"
                    )
                materialized = pool.decode(status, batch_payload, worker)
                saw_materialized = True
                continue
            if saw_acknowledgement:
                raise RuntimeError("worker sample payload followed a batch acknowledgement")
            samples.append(pool.decode(status, payload, worker))
        if saw_materialized:
            return materialized
        if saw_acknowledgement:
            raise RuntimeError("worker batch payload is missing")
        return self._loader._collate_batch(samples)

    def _next_completion(self, expected_position: int) -> tuple[int, bytes, int]:
        pool = self._loader._process_pool
        deadline = pool.deadline()
        while True:
            self._fill_frontier()
            position = self._schedule.try_commit()
            if position is not None:
                if position != expected_position:
                    raise RuntimeError("scheduler committed a noncontiguous position")
                status, payload, worker = self._ready.pop(position)
                self._fill_frontier()
                return status, payload, worker
            progressed = self._poll_completions()
            if not progressed:
                pool.check_workers(deadline)
                time.sleep(POLL_SECONDS)

    def _fill_frontier(self) -> None:
        pool = self._loader._process_pool
        while (dispatch := self._schedule.next_dispatch()) is not None:
            position, worker = dispatch
            index = self._loader._plan.index(
                self._loader.root_seed, self._epoch, position
            )
            batch_size = pool.batch_size
            batch_end = batch_size is None or (
                (position + 1) % batch_size == 0 or position + 1 == self._length
            )
            if not pool.try_submit(
                self._epoch,
                position,
                index,
                worker,
                batch_end=batch_end,
            ):
                return
            self._schedule.mark_dispatched(position, worker)

    def _poll_completions(self) -> bool:
        pool = self._loader._process_pool
        progressed = False
        for worker in range(pool.worker_count):
            completion = pool.try_receive(worker)
            if completion is None:
                continue
            position, status, payload = completion
            self._schedule.mark_completed(position, worker)
            self._ready[position] = (status, payload, worker)
            progressed = True
        return progressed

    def _finish_epoch(self) -> None:
        if not self._complete:
            self._loader._epoch += 1
            self._complete = True

    @property
    def complete(self) -> bool:
        """Report whether exhaustion advanced the loader epoch."""
        return self._complete

    def invalidate(self) -> None:
        """Prevent a replaced iterator from consuming a new pool's completions."""
        self._valid = False

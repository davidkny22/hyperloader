"""Completion-order batch readiness and delivery."""

from __future__ import annotations

import time
from collections import deque
from typing import Any


class CompletionBatchDelivery:
    """Group ready process completions without changing batch composition."""

    def __init__(self, iterator: Any) -> None:
        self._owner = iterator
        self._ready_batches: deque[int] = deque()
        self._queued_batches: set[int] = set()

    def observe(self, position: int) -> None:
        """Queue a batch when this completion closes its final sample gap."""
        owner = self._owner
        width = owner._loader.batch_size or 1
        ordinal = position if owner._worker_batches else position // width
        if ordinal in self._queued_batches:
            return
        if not owner._worker_batches:
            start = ordinal * width
            stop = min(start + width, owner._length)
            if any(sample not in owner._ready for sample in range(start, stop)):
                return
        self._queued_batches.add(ordinal)
        self._ready_batches.append(ordinal)

    def next_batch(self) -> tuple[int, Any, int]:
        """Wait for and deliver the next fully ready batch."""
        owner = self._owner
        pool = owner._loader._process_pool
        deadline = pool.deadline()
        while True:
            owner._fill_frontier()
            if self._ready_batches:
                return self._take_ready_batch()
            progressed = owner._poll_completions()
            if not progressed:
                pool.check_workers(deadline)
                wait_started = time.perf_counter_ns()
                pool.wait_for_completion(deadline)
                owner._schedule.record_wait(time.perf_counter_ns() - wait_started)
                if owner._delivery_telemetry is not None:
                    owner._delivery_telemetry.record_stall()

    def _take_ready_batch(self) -> tuple[int, Any, int]:
        owner = self._owner
        pool = owner._loader._process_pool
        ordinal = self._ready_batches.popleft()
        self._queued_batches.remove(ordinal)
        batch_size = owner._loader.batch_size
        width = batch_size or 1
        if owner._worker_batches:
            status, payload, worker = owner._ready.pop(ordinal)
            if owner._schedule.try_commit_ready(ordinal) != ordinal:
                raise RuntimeError("ready batch could not commit")
            value = pool.decode_batch(status, payload, worker)
            delivered = min(width, owner._length - ordinal * width)
        else:
            start = ordinal * width
            stop = min(start + width, owner._length)
            values = []
            for position in range(start, stop):
                status, payload, worker = owner._ready.pop(position)
                if owner._schedule.try_commit_ready(position) != position:
                    raise RuntimeError("ready sample could not commit")
                values.append(pool.decode(status, payload, worker))
            value = (
                values[0]
                if batch_size is None
                else owner._loader._collate_batch(values)
            )
            delivered = stop - start
        owner._fill_frontier()
        return ordinal, value, delivered

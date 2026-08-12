"""Owner-side prefetch and delivery over native Torch-compatible lanes."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .lane_pool import CompatLanePool
from .protocol import LaneExhausted, TaggedBatch


class CompatLaneRuntime(Iterator[Any]):
    """Dispatch Torch sampler fetch units and deliver native completions."""

    def __init__(
        self,
        loader: Any,
        pool: CompatLanePool,
        *,
        skip: int = 0,
        phase: int = 0,
        prefetch: int | None = None,
    ) -> None:
        self._loader = loader
        self._pool = pool
        self._sampler = iter(loader._compat_reference._index_sampler)
        for _ in range(skip):
            try:
                next(self._sampler)
            except StopIteration:
                break
        self._start_task = skip
        self._next_task = skip
        self._next_delivery = skip
        self._phase = phase
        self._iterable = loader._plan is None
        self._active_workers = set(range(loader.num_workers))
        self._worker_cursor = phase
        prefetch = loader._compat_reference.prefetch_factor or 2
        self._limit = loader.num_workers * prefetch
        self._ready: dict[int, Any] = {}
        self._sampler_exhausted = False
        self._closed = False
        self._fill(self._limit if prefetch is None else prefetch)

    def __iter__(self) -> CompatLaneRuntime:
        return self

    def __next__(self) -> Any:
        if self._closed:
            raise StopIteration
        deadline = self._pool.deadline()
        while True:
            self._fill()
            if self._loader.in_order:
                if self._next_delivery in self._ready:
                    value = self._ready.pop(self._next_delivery)
                    self._next_delivery += 1
                    self._fill()
                    if isinstance(value, LaneExhausted):
                        continue
                    return value
            elif self._ready:
                task = next(iter(self._ready))
                value = self._ready.pop(task)
                self._next_delivery += 1
                self._fill()
                if isinstance(value, LaneExhausted):
                    continue
                return value
            if (
                self._sampler_exhausted
                and not self._pool.has_pending
                and not self._ready
            ):
                self._closed = True
                if not self._loader.persistent_workers:
                    self._pool.close()
                    self._loader._compat_lane_pool = None
                raise StopIteration
            completion = self._pool.try_receive()
            if completion is None:
                self._pool.wait_for_completion(deadline)
                continue
            task, value = completion
            self._accept(task, value)

    def _fill(self, target: int | None = None) -> None:
        limit = self._limit if target is None else target
        while (
            not self._sampler_exhausted
            and len(self._ready) + self._pool.pending_count < limit
        ):
            try:
                indices = next(self._sampler)
            except StopIteration:
                self._sampler_exhausted = True
                return
            worker = self._next_worker()
            if worker is None:
                self._sampler_exhausted = True
                return
            if not self._pool.try_submit(self._next_task, indices, worker):
                return
            self._next_task += 1

    def capture_points(self) -> tuple[TaggedBatch, ...]:
        """Retain the first undelivered snapshot already admitted per lane."""
        points = self._snapshot_points()
        targets = set(self._pool.pending_workers) | set(points)
        deadline = self._pool.deadline()
        while not targets.issubset(points) and self._pool.has_pending:
            completion = self._pool.try_receive()
            if completion is None:
                self._pool.wait_for_completion(deadline)
                continue
            task, value = completion
            self._accept(task, value)
            if isinstance(value, TaggedBatch):
                points.setdefault(value.worker, value)
        return tuple(points[worker] for worker in sorted(points))

    def _snapshot_points(self) -> dict[int, TaggedBatch]:
        points: dict[int, TaggedBatch] = {}
        for task in sorted(self._ready):
            value = self._ready[task]
            if isinstance(value, TaggedBatch):
                points.setdefault(value.worker, value)
        return points

    def _accept(self, task: int, value: Any) -> None:
        self._ready[task] = value
        if not isinstance(value, LaneExhausted):
            return
        self._active_workers.discard(value.worker)
        if not self._active_workers:
            self._sampler_exhausted = True

    def _next_worker(self) -> int | None:
        if not self._iterable:
            return (
                self._phase + self._next_task - self._start_task
            ) % self._loader.num_workers
        for _ in range(self._loader.num_workers):
            worker = self._worker_cursor % self._loader.num_workers
            self._worker_cursor += 1
            if worker in self._active_workers:
                return worker
        return None

    @property
    def sampler_position(self) -> int:
        """Return the number of fetch units admitted from the sampler."""
        return self._next_task

    def _shutdown_workers(self) -> None:
        """Close nonpersistent native lanes when an iterator is invalidated."""
        self._closed = True
        self._pool.close()

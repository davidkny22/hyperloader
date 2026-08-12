"""Deterministic round-robin delivery from logical iterable lanes."""

from __future__ import annotations

import pickle
from collections import deque
from collections.abc import Iterator
from typing import Any

from .factory import logical_lane_count
from .lane import IterableLane
from .worker_info import lane_worker_info


class IterableIterator(Iterator[Any]):
    """Deliver lane-whole batches from a fixed logical lane set."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self._epoch = loader._epoch
        self._delivered = 0
        self._complete = False
        self._valid = True
        self._lane_count = logical_lane_count(loader)
        self._lanes = deque(self._build_lane(lane) for lane in range(self._lane_count))

    def __iter__(self) -> IterableIterator:
        return self

    def __next__(self) -> Any:
        if not self._valid:
            raise RuntimeError("iterable lane iterator is no longer active")
        while self._lanes:
            lane = self._lanes.popleft()
            values, exhausted = self._next_lane_batch(lane)
            if not exhausted:
                self._lanes.append(lane)
            if not values:
                continue
            if (
                exhausted
                and self._loader.batch_size is not None
                and len(values) < self._loader.batch_size
                and self._loader.drop_last
            ):
                continue
            self._delivered += 1
            self._loader._epoch_state.mark_delivered(self._epoch)
            return (
                values[0]
                if self._loader.batch_size is None
                else self._loader._collate_batch(values)
            )
        self._finish_epoch()
        raise StopIteration

    @property
    def complete(self) -> bool:
        """Report whether every logical lane exhausted."""
        return self._complete

    @property
    def coordinate_epoch(self) -> int:
        """Return the epoch owned by this lane set."""
        return self._epoch

    @property
    def delivered_batches(self) -> int:
        """Return the lane-whole delivered-batch prefix."""
        return self._delivered

    @property
    def sampler_checksum(self) -> int:
        """Return the native iterable checksum sentinel."""
        return 0

    @property
    def delivered_bitmap(self) -> bytes:
        """Return the strict iterable delivery sentinel."""
        return b""

    def invalidate(self) -> None:
        """Prevent a replaced lane set from producing more values."""
        self._valid = False
        self._lanes.clear()

    def _build_lane(self, identity: int) -> IterableLane:
        payload = self._loader._iterable_payload
        dataset = self._loader.dataset if payload is None else pickle.loads(payload)
        with lane_worker_info(identity, self._lane_count, dataset, None):
            if self._loader.worker_init_fn is not None:
                self._loader.worker_init_fn(identity)
            iterator = iter(dataset)
        return IterableLane(identity, dataset, iterator)

    def _next_lane_batch(self, lane: IterableLane) -> tuple[list[Any], bool]:
        width = self._loader.batch_size or 1
        values = []
        exhausted = False
        while len(values) < width:
            with lane_worker_info(
                lane.identity,
                self._lane_count,
                lane.dataset,
                None,
            ):
                try:
                    value = next(lane.iterator)
                except StopIteration:
                    exhausted = True
                    break
            values.append(value)
            lane.arrival += 1
        return values, exhausted

    def _finish_epoch(self) -> None:
        if self._complete:
            return
        self._loader._epoch_state.complete(self._epoch)
        self._complete = True

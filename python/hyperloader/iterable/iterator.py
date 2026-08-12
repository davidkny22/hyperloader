"""Deterministic round-robin delivery from logical iterable lanes."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .factory import logical_lane_count
from .runtime import IterableLaneRuntime
from .state import IterableCheckpoint


class IterableIterator(Iterator[Any]):
    """Deliver lane-whole batches from a fixed logical lane set."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self._epoch = loader._epoch
        self._delivered = 0
        self._complete = False
        self._valid = True
        self._lane_count = logical_lane_count(loader)
        self._runtime = IterableLaneRuntime(loader, self._epoch, self._lane_count)
        self._lanes = self._runtime.build_lanes(loader._resume_iterable_state)

    def __iter__(self) -> IterableIterator:
        return self

    def __next__(self) -> Any:
        if not self._valid:
            raise RuntimeError("iterable lane iterator is no longer active")
        while self._lanes:
            lane = self._lanes.popleft()
            values, exhausted = self._runtime.next_batch(lane)
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
            self._runtime.mark_delivered(lane)
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

    def capture_checkpoint(self) -> IterableCheckpoint:
        """Read selected snapshots without calling back into a live source."""
        order = tuple(lane.identity for lane in self._lanes)
        return self._runtime.capture_checkpoint(order)

    def recover_lane(self, identity: int) -> None:
        """Rebuild one failed lane from the engine-owned delivered checkpoint."""
        order = tuple(lane.identity for lane in self._lanes)
        self._lanes = self._runtime.recover_lane(identity, order)

    def invalidate(self) -> None:
        """Prevent a replaced lane set from producing more values."""
        self._valid = False
        self._lanes.clear()

    def _finish_epoch(self) -> None:
        if self._complete:
            return
        self._loader._epoch_state.complete(self._epoch)
        self._complete = True

"""Bounded engine-owned source snapshot rings."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Retain one serialized source state and its arrival boundary."""

    arrival: int
    payload: bytes


class SnapshotRing:
    """Retain bounded source states and select a delivered-prefix restore point."""

    def __init__(
        self,
        *,
        stateful: bool,
        cadence: int | str,
        maximum_bytes: int,
        depth: int,
    ) -> None:
        self.stateful = stateful
        self.cadence = cadence
        self.maximum_bytes = maximum_bytes
        self._entries: deque[SourceSnapshot] = deque(maxlen=max(1, depth))
        self.snapshotless = cadence == "off" or not stateful

    def due(self, produced_batches: int, *, force: bool = False) -> bool:
        """Return whether this production boundary should capture state."""
        if self.snapshotless:
            return False
        return force or produced_batches % int(self.cadence) == 0

    def push(self, arrival: int, payload: bytes) -> bool:
        """Push a state or degrade this lane when it exceeds the byte cap."""
        if len(payload) > self.maximum_bytes:
            self.snapshotless = True
            self._entries.clear()
            return False
        self._entries.append(SourceSnapshot(arrival, payload))
        return True

    def seed(self, snapshot: SourceSnapshot) -> None:
        """Seed a restored ring with its selected checkpoint entry."""
        if not self.snapshotless:
            self._entries.append(snapshot)

    def select(self, delivered_arrival: int) -> SourceSnapshot | None:
        """Return the newest retained state at or below the delivered frontier."""
        selected = None
        for entry in self._entries:
            if entry.arrival <= delivered_arrival:
                selected = entry
        return selected

    def discard_before(self, delivered_arrival: int) -> None:
        """Discard obsolete states while retaining the newest delivered anchor."""
        selected = self.select(delivered_arrival)
        future = [
            entry for entry in self._entries if entry.arrival > delivered_arrival
        ]
        self._entries.clear()
        if selected is not None:
            self._entries.append(selected)
        self._entries.extend(future)

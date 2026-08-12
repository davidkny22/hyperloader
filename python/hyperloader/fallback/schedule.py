"""Pure-Python bounded frontier with exact native transition rules."""

from __future__ import annotations


class StaticSchedule:
    """Track dispatch, completion, and delivery in a bounded position window."""

    def __init__(self, start: int, end: int, depth: int, worker_count: int) -> None:
        if end < start:
            raise ValueError("schedule end precedes its start")
        if depth <= 0:
            raise ValueError("frontier depth must be positive")
        if worker_count <= 0:
            raise ValueError("worker count must be positive")
        self._end = end
        self._depth = depth
        self._worker_count = worker_count
        self._worker_ceiling = worker_count
        self._next_commit = start
        self._dispatch_ordinal = 0
        self._positions: dict[int, tuple[str, int]] = {}
        self._delivered: set[int] = set()

    def next_dispatch(self) -> tuple[int, int] | None:
        """Return the next FIFO dispatch candidate."""
        candidates = self.dispatch_candidates()
        return None if not candidates else self.dispatch_at(candidates[0])

    def dispatch_candidates(self) -> list[int]:
        """Return unsubmitted positions admitted by the frontier."""
        stop = min(self._end, self._next_commit + self._depth)
        return [
            position
            for position in range(self._next_commit, stop)
            if position not in self._positions and position not in self._delivered
        ]

    def dispatch_at(self, position: int) -> tuple[int, int] | None:
        """Return the current route for one eligible position."""
        stop = min(self._end, self._next_commit + self._depth)
        if (
            position < self._next_commit
            or position >= stop
            or position in self._positions
            or position in self._delivered
        ):
            return None
        return position, self._dispatch_ordinal % self._worker_count

    def mark_dispatched(self, position: int, worker: int) -> None:
        """Confirm that one candidate entered its worker transport."""
        if self.dispatch_at(position) != (position, worker):
            raise ValueError("dispatch is not an eligible frontier candidate")
        self._positions[position] = ("dispatched", worker)
        self._dispatch_ordinal += 1

    def mark_completed(self, position: int, worker: int) -> None:
        """Record an out-of-order completion."""
        state = self._positions.get(position)
        if state is None:
            raise ValueError("completion is outside the active frontier")
        if state[0] == "ready":
            raise ValueError("position completed twice")
        if state[1] != worker:
            raise ValueError("completion came from the wrong worker")
        self._positions[position] = ("ready", worker)

    def try_commit(self) -> int | None:
        """Commit the next stream position only when it is ready."""
        return self.try_commit_ready(self._next_commit)

    def try_commit_ready(self, position: int) -> int | None:
        """Commit one ready position and advance the contiguous base."""
        if self._positions.get(position, (None,))[0] != "ready":
            return None
        del self._positions[position]
        if position == self._next_commit:
            self._next_commit += 1
            while self._next_commit in self._delivered:
                self._delivered.remove(self._next_commit)
                self._next_commit += 1
        else:
            self._delivered.add(position)
        return position

    def delivered_positions(self) -> list[int]:
        """Return committed positions beyond the contiguous prefix."""
        return sorted(self._delivered)

    def seed_delivered(self, position: int) -> None:
        """Seed one restored completion beyond the contiguous prefix."""
        if position <= self._next_commit or position >= self._end:
            raise ValueError("restored delivery is outside the open position range")
        if position in self._positions or position in self._delivered:
            raise ValueError("restored position was delivered twice")
        self._delivered.add(position)

    def is_complete(self) -> bool:
        """Return whether every position has committed."""
        return (
            self._next_commit == self._end
            and not self._positions
            and not self._delivered
        )

    def occupied(self) -> int:
        """Return dispatched positions not yet committed."""
        return len(self._positions)

    def set_depth(self, depth: int) -> None:
        """Change frontier depth without evicting active positions."""
        if depth <= 0:
            raise ValueError("frontier depth must be positive")
        stop = self._next_commit + depth
        if any(position >= stop for position in self._positions):
            raise ValueError("frontier depth excludes an active position")
        self._depth = depth

    def set_worker_count(self, worker_count: int) -> None:
        """Change live routing width within the spawned ceiling."""
        if worker_count <= 0 or worker_count > self._worker_ceiling:
            raise ValueError("live worker count is outside the plan ceiling")
        self._worker_count = worker_count
        self._dispatch_ordinal %= worker_count

"""Adaptive frontier runtime and focused scheduling observations."""

from __future__ import annotations

from typing import Any

from hyperloader import _hyperloader
from hyperloader.config import AUTO


def binding_cause(loader: Any) -> str:
    """Name the input that selected the active frontier regime."""
    pool = loader._process_pool
    if pool.frontier_budget_bound:
        return "memory-budget"
    scheduler = loader.config.scheduler
    if scheduler.frontier_depth is not AUTO:
        return "explicit-depth"
    if loader.prefetch_factor is not AUTO and loader.prefetch_factor is not None:
        return "prefetch-hint"
    profile = loader._cost_profile
    if profile is None or profile.statistics() is None:
        return "cold-variance"
    return "profile-tail"


class FrontierRuntime:
    """Coordinate native scheduling while growing monotonically on a stall."""

    def __init__(
        self,
        end: int,
        depth: int,
        ceiling: int,
        worker_count: int,
        growth_multiplier: int,
        binding: str,
    ) -> None:
        self._schedule = _hyperloader._StaticSchedule(0, end, depth, worker_count)
        self._initial_depth = depth
        self._depth = depth
        self._ceiling = ceiling
        self._growth_multiplier = growth_multiplier
        self._binding = binding
        self._max_occupied = 0
        self._growth_events = 0
        self._wait_ns = 0
        self._active_ns = 0

    def next_dispatch(self) -> tuple[int, int] | None:
        """Return the next native dispatch candidate."""
        return self._schedule.next_dispatch()

    def mark_dispatched(self, position: int, worker: int) -> None:
        """Commit one transport admission and update peak occupancy."""
        self._schedule.mark_dispatched(position, worker)
        self._max_occupied = max(self._max_occupied, self._schedule.occupied())

    def mark_completed(self, position: int, worker: int) -> None:
        """Record one out-of-order completion."""
        self._schedule.mark_completed(position, worker)

    def try_commit(self) -> int | None:
        """Commit only the next sampler-order position."""
        return self._schedule.try_commit()

    def record_wait(self, wait_ns: int) -> None:
        """Record a delivery stall and grow a saturated frontier within its ceiling."""
        self._wait_ns += wait_ns
        if self._depth >= self._ceiling or self._schedule.occupied() < self._depth:
            return
        grown = min(self._ceiling, self._depth * self._growth_multiplier)
        if grown == self._depth:
            return
        self._schedule.set_depth(grown)
        self._depth = grown
        self._growth_events += 1

    def record_active(self, active_ns: int) -> None:
        """Accumulate loader execution time without consumer work."""
        self._active_ns += active_ns

    def report(self) -> dict[str, int | float | str]:
        """Return the focused scheduler evidence consumed by the gate harness."""
        stall_fraction = self._wait_ns / self._active_ns if self._active_ns else 0.0
        return {
            "active_ns": self._active_ns,
            "binding": self._binding,
            "ceiling": self._ceiling,
            "final_depth": self._depth,
            "growth_events": self._growth_events,
            "initial_depth": self._initial_depth,
            "max_occupied": self._max_occupied,
            "stall_fraction": stall_fraction,
            "wait_ns": self._wait_ns,
        }

    @property
    def occupied(self) -> int:
        """Return current dispatched, uncommitted occupancy."""
        return self._schedule.occupied()

"""Adaptive frontier runtime and focused scheduling observations."""

from __future__ import annotations

from collections.abc import Callable
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
        cost_estimator: Callable[[int], float | None] | None = None,
        *,
        start: int = 0,
    ) -> None:
        self._schedule = _hyperloader._StaticSchedule(start, end, depth, worker_count)
        self._initial_depth = depth
        self._depth = depth
        self._ceiling = ceiling
        self._growth_multiplier = growth_multiplier
        self._binding = binding
        self._cost_estimator = cost_estimator
        self._max_occupied = 0
        self._growth_events = 0
        self._wait_ns = 0
        self._active_ns = 0
        self._stalled_since_delivery = False

    def next_dispatch(self) -> tuple[int, int] | None:
        """Return the next native dispatch candidate."""
        order = self.dispatch_order()
        return self.dispatch_at(order[0]) if order else None

    def dispatch_order(self) -> list[int]:
        """Order the admitted window by descending known cost, then position."""
        candidates = self._schedule.dispatch_candidates()
        if self._cost_estimator is None:
            return candidates
        estimates = {
            position: self._cost_estimator(position) for position in candidates
        }
        if not any(estimate is not None for estimate in estimates.values()):
            return candidates
        return sorted(
            candidates,
            key=lambda position: (
                estimates[position] is None,
                -(estimates[position] or 0.0),
                position,
            ),
        )

    def dispatch_at(self, position: int) -> tuple[int, int] | None:
        """Route one selected position when it remains frontier-eligible."""
        return self._schedule.dispatch_at(position)

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
        self._stalled_since_delivery = True
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

    def set_worker_count(self, worker_count: int) -> None:
        """Park or unpark scheduler routes within the spawned worker ceiling."""
        self._schedule.set_worker_count(worker_count)

    def consume_stall_flag(self) -> bool:
        """Return and clear whether delivery waited since the previous batch."""
        stalled = self._stalled_since_delivery
        self._stalled_since_delivery = False
        return stalled

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

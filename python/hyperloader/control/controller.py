"""Cadenced width adaptation with clipping and shrink hysteresis."""

from __future__ import annotations

from dataclasses import dataclass

from .objective import ControllerObjective


@dataclass(frozen=True, slots=True)
class ControllerDecision:
    """One auditable live-width decision."""

    previous_width: int
    width: int
    reason: str
    starvation: bool
    score: tuple[int, float]
    binding: str | None = None


class AdaptiveController:
    """Keep starvation at zero, then release unnecessary worker routes."""

    def __init__(
        self,
        *,
        width_ceiling: int,
        cadence_seconds: float,
        cadence_batches: int,
        step_clip: int,
        shrink_hysteresis: int,
        objective: ControllerObjective,
        cpu_ceiling_binding: bool = False,
        bandwidth_ceiling: float | None = None,
        work_shape: str = "compute",
        cluster: str = "all",
    ) -> None:
        if width_ceiling <= 0 or cadence_seconds <= 0 or cadence_batches <= 0:
            raise ValueError("controller ceiling and cadence must be positive")
        if step_clip <= 0 or shrink_hysteresis <= 0:
            raise ValueError("controller clipping and hysteresis must be positive")
        if bandwidth_ceiling is not None and bandwidth_ceiling < 0:
            raise ValueError("controller bandwidth ceiling must be nonnegative")
        self.width_ceiling = width_ceiling
        self.width = width_ceiling
        self._cadence_ns = int(cadence_seconds * 1_000_000_000)
        self._cadence_batches = cadence_batches
        self._step_clip = step_clip
        self._shrink_hysteresis = shrink_hysteresis
        self._objective = objective
        self._cpu_ceiling_binding = cpu_ceiling_binding
        self._bandwidth_ceiling = bandwidth_ceiling
        self._work_shape = work_shape
        self._cluster = cluster
        self._last_decision_ns: int | None = None
        self._batches = 0
        self._stalled = False
        self._shrink_cadences = 0
        self.decisions: list[ControllerDecision] = []

    def observe(
        self,
        *,
        now_ns: int,
        stalled: bool,
        occupied: int,
        batch_size: int,
        bytes_per_second: float = 0.0,
    ) -> ControllerDecision | None:
        """Consume one delivery observation and decide only at cadence."""
        self._last_decision_ns = self._last_decision_ns or now_ns
        self._batches += 1
        self._stalled = self._stalled or stalled
        elapsed = now_ns - self._last_decision_ns
        if self._batches < self._cadence_batches or elapsed < self._cadence_ns:
            return None
        starvation = self._stalled or occupied < batch_size
        previous = self.width
        reason = "hold"
        binding = None
        if (
            self._bandwidth_ceiling is not None
            and bytes_per_second > self._bandwidth_ceiling
        ):
            self.width = max(1, self.width - self._step_clip)
            self._shrink_cadences = 0
            reason = "bandwidth-ceiling"
            binding = "bandwidth"
        elif starvation:
            self.width = min(self.width_ceiling, self.width + self._step_clip)
            self._shrink_cadences = 0
            if self.width == previous and self._cpu_ceiling_binding:
                reason = "cpu-ceiling"
                binding = "cpu_cores"
            else:
                reason = "starvation"
        else:
            self._shrink_cadences += 1
            if self._shrink_cadences >= self._shrink_hysteresis and self.width > 1:
                self.width = max(1, self.width - self._step_clip)
                self._shrink_cadences = 0
                reason = "resource-minimum"
        score = self._objective.score(
            starvation=starvation,
            width=self.width,
            work_shape=self._work_shape,
            cluster=self._cluster,
            bytes_per_second=bytes_per_second,
        )
        decision = ControllerDecision(
            previous, self.width, reason, starvation, score, binding
        )
        self.decisions.append(decision)
        self._last_decision_ns = now_ns
        self._batches = 0
        self._stalled = False
        return decision

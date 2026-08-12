"""Paired comparison records and terminal dominance decisions."""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from benchmark_protocol import EnvironmentMetadata, TuningBudget

REQUIRED_PAIRS = 5
PLATFORM_NOISE_PERCENT = 0.93
BOOTSTRAP_DRAWS = 10_000


class DominanceProtocolError(ValueError):
    """A comparison record violates its fixed protocol."""


@dataclass(frozen=True)
class SelectedConfig:
    """One system's best configuration under the common search budget."""

    workers: int
    prefetch_factor: int


@dataclass(frozen=True)
class DominanceRun:
    """One timed half of a reference comparison."""

    system: str
    reference: str
    workload: str
    gpu_regime: str
    throughput: float
    duration_seconds: float
    warmed: bool
    selected: SelectedConfig
    tuning: TuningBudget
    environment: EnvironmentMetadata


@dataclass(frozen=True)
class DominanceObservation:
    """One alternating hyperloader and reference feeder swap."""

    ordinal: int
    first: DominanceRun
    second: DominanceRun
    uninterrupted: bool

    def run(self, system: str) -> DominanceRun:
        """Return one named half independent of pair order."""
        if self.first.system == system:
            return self.first
        if self.second.system == system:
            return self.second
        raise DominanceProtocolError(f"observation has no {system} run")

    def advantage_percent(self) -> float:
        """Return hyperloader throughput gain relative to its reference."""
        hyperloader = self.run("hyperloader").throughput
        reference = self.run(self.first.reference).throughput
        return 100.0 * (hyperloader - reference) / reference


@dataclass(frozen=True)
class DominanceDecision:
    """Bootstrap interval and terminal win, tie, or loss."""

    status: str
    pairs: int
    mean_advantage_percent: float
    lower_percent: float
    upper_percent: float
    tie_margin_percent: float


def validate_observations(observations: Sequence[DominanceObservation]) -> None:
    """Reject changed inputs, unequal budgets, and invalid pair ordering."""
    if not observations:
        raise DominanceProtocolError("the comparison has no observations")
    expected_ordinals = list(range(len(observations)))
    if [observation.ordinal for observation in observations] != expected_ordinals:
        raise DominanceProtocolError(
            "observation ordinals must be contiguous from zero"
        )
    first = observations[0].first
    selected_by_system = {
        run.system: run.selected
        for run in (observations[0].first, observations[0].second)
    }
    campaign_key = (
        first.reference,
        first.workload,
        first.gpu_regime,
        first.tuning,
        first.environment.stability_key(),
    )
    for observation in observations:
        _validate_pair(observation)
        for run in (observation.first, observation.second):
            if run.selected != selected_by_system[run.system]:
                raise DominanceProtocolError(
                    "a selected system configuration changed between cells"
                )
            key = (
                run.reference,
                run.workload,
                run.gpu_regime,
                run.tuning,
                run.environment.stability_key(),
            )
            if key != campaign_key:
                raise DominanceProtocolError(
                    "comparison controls changed between cells"
                )


def decide(
    observations: Sequence[DominanceObservation],
    *,
    seed: int = 0,
    draws: int = BOOTSTRAP_DRAWS,
) -> DominanceDecision:
    """Classify a win or noise-bounded tie after five pairs."""
    validate_observations(observations)
    if len(observations) < REQUIRED_PAIRS:
        raise DominanceProtocolError("a decision requires five paired cells")
    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    values = [observation.advantage_percent() for observation in observations]
    lower, upper = _bootstrap_interval(values, seed=seed, draws=draws)
    if lower > 0:
        status = "win"
    elif lower >= -PLATFORM_NOISE_PERCENT:
        status = "tie"
    else:
        status = "loss"
    return DominanceDecision(
        status=status,
        pairs=len(values),
        mean_advantage_percent=statistics.fmean(values),
        lower_percent=lower,
        upper_percent=upper,
        tie_margin_percent=PLATFORM_NOISE_PERCENT,
    )


def _validate_pair(observation: DominanceObservation) -> None:
    reference = observation.first.reference
    expected_first = "hyperloader" if observation.ordinal % 2 == 0 else reference
    if observation.first.system != expected_first:
        raise DominanceProtocolError("pair order must alternate by ordinal")
    if {observation.first.system, observation.second.system} != {
        "hyperloader",
        reference,
    }:
        raise DominanceProtocolError("each pair needs hyperloader and one reference")
    if reference not in {"torch", "spdl"}:
        raise DominanceProtocolError("reference must be torch or spdl")
    if not observation.uninterrupted:
        raise DominanceProtocolError(
            "GPU work must remain uninterrupted across the swap"
        )
    if observation.first.tuning != observation.second.tuning:
        raise DominanceProtocolError("both systems need the same counted tuning budget")
    if observation.first.environment != observation.second.environment:
        raise DominanceProtocolError("both halves need identical environment metadata")
    for run in (observation.first, observation.second):
        if run.duration_seconds != 45.0:
            raise DominanceProtocolError("each definitive half must last 45 seconds")
        if not run.warmed or run.throughput <= 0:
            raise DominanceProtocolError(
                "each half must be warm with positive throughput"
            )


def _bootstrap_interval(
    values: list[float], *, seed: int, draws: int
) -> tuple[float, float]:
    generator = random.Random(seed)
    count = len(values)
    means = sorted(
        statistics.fmean(generator.choice(values) for _ in range(count))
        for _ in range(draws)
    )
    return _quantile(means, 0.025), _quantile(means, 0.975)


def _quantile(values: list[float], probability: float) -> float:
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction

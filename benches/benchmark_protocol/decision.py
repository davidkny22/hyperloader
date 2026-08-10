"""Bootstrap interval and terminal paired-campaign decision."""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from .models import PairedObservation
from .validation import validate_observations

MIN_PAIRS = 10
MAX_PAIRS = 40
MAX_HALF_WIDTH_PERCENT = 0.15
BOOTSTRAP_DRAWS = 10_000


@dataclass(frozen=True)
class DecisionResult:
    """Measured paired penalty, uncertainty, and collection decision."""

    status: str
    pairs: int
    mean_penalty_percent: float
    lower_percent: float
    upper_percent: float
    half_width_percent: float
    threshold_percent: float


def evaluate(
    observations: Sequence[PairedObservation],
    *,
    threshold_percent: float,
    seed: int = 0,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> DecisionResult:
    """Apply the pinned replication, precision, cap, and upper-bound rule."""
    validate_observations(observations)
    if threshold_percent < 0:
        raise ValueError("decision threshold must be nonnegative")
    if bootstrap_draws <= 0:
        raise ValueError("bootstrap draw count must be positive")
    penalties = [observation.penalty_percent() for observation in observations]
    mean = statistics.fmean(penalties)
    lower, upper = _bootstrap_interval(penalties, seed, bootstrap_draws)
    half_width = (upper - lower) / 2.0
    count = len(penalties)
    if count < MIN_PAIRS:
        status = "collect"
    elif half_width > MAX_HALF_WIDTH_PERCENT and count < MAX_PAIRS:
        status = "collect"
    else:
        status = "pass" if upper < threshold_percent else "fail"
    return DecisionResult(
        status=status,
        pairs=count,
        mean_penalty_percent=mean,
        lower_percent=lower,
        upper_percent=upper,
        half_width_percent=half_width,
        threshold_percent=threshold_percent,
    )


def _bootstrap_interval(
    values: list[float], seed: int, draws: int
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

"""Bootstrap decision for one live-training comparison point."""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from .models import TrainingObservation
from .validation import validate_observations


@dataclass(frozen=True)
class TrainingDecision:
    """Loader tax interval and collection disposition for one point."""

    status: str
    pairs: int
    mean_tax_percent: float
    lower_percent: float
    upper_percent: float
    half_width_percent: float
    threshold_percent: float
    mode: str


def decide(observations: Sequence[TrainingObservation]) -> TrainingDecision:
    """Apply the point's preregistered replication, precision, and interval rule."""
    validate_observations(observations)
    rule = observations[0].config.decision
    values = [observation.tax_percent() for observation in observations]
    lower, upper = _bootstrap_interval(
        values,
        seed=rule.bootstrap_seed,
        draws=rule.bootstrap_draws,
    )
    half_width = (upper - lower) / 2.0
    count = len(values)
    needs_more_precision = (
        half_width > rule.max_half_width_percent and count < rule.max_pairs
    )
    if count < rule.min_pairs or needs_more_precision:
        status = "collect"
    elif rule.mode == "upper":
        status = "pass" if upper < rule.threshold_percent else "fail"
    else:
        bound = max(abs(lower), abs(upper))
        status = "pass" if bound < rule.threshold_percent else "fail"
    return TrainingDecision(
        status=status,
        pairs=count,
        mean_tax_percent=statistics.fmean(values),
        lower_percent=lower,
        upper_percent=upper,
        half_width_percent=half_width,
        threshold_percent=rule.threshold_percent,
        mode=rule.mode,
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

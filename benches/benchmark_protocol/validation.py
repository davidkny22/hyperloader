"""Protocol invariants that reject incomparable benchmark records."""

from __future__ import annotations

from collections.abc import Sequence

from .matrix import workload_names
from .models import PairedObservation, SystemRun


class ProtocolError(ValueError):
    """A benchmark record violates a comparison invariant."""


def validate_observations(observations: Sequence[PairedObservation]) -> None:
    """Reject invalid pair order, tuning, metadata, and workload inputs."""
    if not observations:
        raise ProtocolError("the paired campaign has no observations")
    ordered = sorted(observations, key=lambda observation: observation.ordinal)
    if list(observations) != ordered:
        raise ProtocolError("paired observations must be ordered by ordinal")
    if [observation.ordinal for observation in ordered] != list(range(len(ordered))):
        raise ProtocolError("paired observation ordinals must be contiguous from zero")

    campaign_key = ordered[0].first.environment.stability_key()
    campaign_config = ordered[0].first.config
    campaign_tuning = ordered[0].first.tuning
    for observation in ordered:
        _validate_pair(observation)
        if observation.first.environment.stability_key() != campaign_key:
            raise ProtocolError("campaign environment controls changed between cells")
        if observation.first.config != campaign_config:
            raise ProtocolError("one decision campaign cannot mix common configurations")
        if observation.first.tuning != campaign_tuning:
            raise ProtocolError("one decision campaign cannot change its tuning budget")


def _validate_pair(observation: PairedObservation) -> None:
    expected_first = "counterfactual" if observation.ordinal % 2 == 0 else "loader"
    if observation.first.system != expected_first:
        raise ProtocolError("pair order must alternate by observation ordinal")
    if {observation.first.system, observation.second.system} != {
        "counterfactual",
        "loader",
    }:
        raise ProtocolError("each pair needs one loader and one counterfactual run")
    if not observation.uninterrupted:
        raise ProtocolError("the workload must remain uninterrupted across the feeder swap")
    if observation.first.config != observation.second.config:
        raise ProtocolError("both systems must use identical common configuration")
    if observation.first.tuning != observation.second.tuning:
        raise ProtocolError("both systems must receive equal counted tuning budgets")
    if observation.first.environment != observation.second.environment:
        raise ProtocolError("both halves of a pair must share captured environment metadata")
    for run in (observation.first, observation.second):
        _validate_run(run)


def _validate_run(run: SystemRun) -> None:
    if run.config.workload not in workload_names():
        raise ProtocolError(f"unknown workload cell {run.config.workload}")
    if run.config.gpu_regime not in {"compute", "bandwidth"}:
        raise ProtocolError("GPU regime must be compute or bandwidth")
    if run.config.cache_regime not in {"cold", "warm"}:
        raise ProtocolError("cache regime must be cold or warm")
    if run.config.cache_regime != run.environment.cache_regime:
        raise ProtocolError("configuration and environment cache regimes differ")
    if run.duration_seconds != 45.0:
        raise ProtocolError("each half of the 90-second cell must last 45 seconds")
    if not run.warmed:
        raise ProtocolError("both feeders must be pre-warmed outside timing")
    if run.throughput <= 0:
        raise ProtocolError("throughput must be positive")
    if run.config.batch_size <= 0 or run.config.workers < 0:
        raise ProtocolError("batch size and worker count are invalid")
    zero_tuning = (
        run.tuning.trials == 0
        and run.tuning.wall_seconds == 0
        and not run.tuning.knobs
    )
    positive_tuning = (
        run.tuning.trials > 0
        and run.tuning.wall_seconds > 0
        and bool(run.tuning.knobs)
    )
    if not zero_tuning and not positive_tuning:
        raise ProtocolError("tuning budget must be exact zero or fully positive")
    if not run.environment.benchmark_mode or run.environment.concurrent_load:
        raise ProtocolError("benchmark mode requires an exclusive measurement window")
    if not run.environment.cpu_governor or not run.environment.gpu_clock:
        raise ProtocolError("clock and governor observations are required")

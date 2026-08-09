"""Immutable records for paired benchmark capture."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentMetadata:
    """Machine identity and benchmark-mode controls for one run."""

    captured_at: str
    machine: str
    operating_system: str
    kernel: str
    architecture: str
    python: str
    commit: str
    cpu_governor: str
    gpu_clock: str
    cache_regime: str
    benchmark_mode: bool
    concurrent_load: bool

    def stability_key(self) -> tuple[object, ...]:
        """Return fields that must remain fixed across one campaign."""
        return (
            self.machine,
            self.operating_system,
            self.kernel,
            self.architecture,
            self.python,
            self.commit,
            self.cpu_governor,
            self.gpu_clock,
            self.cache_regime,
            self.benchmark_mode,
            self.concurrent_load,
        )


@dataclass(frozen=True)
class CommonConfig:
    """Inputs that must be identical on both sides of a pair."""

    workload: str
    gpu_regime: str
    batch_size: int
    workers: int
    prefetch_depth: int
    delivery: str
    batch_shape: str
    cache_regime: str


@dataclass(frozen=True)
class TuningBudget:
    """Counted search allowance shared by both compared systems."""

    trials: int
    wall_seconds: float
    knobs: tuple[str, ...]


@dataclass(frozen=True)
class SystemRun:
    """One half of a continuous paired cell."""

    system: str
    throughput: float
    duration_seconds: float
    warmed: bool
    config: CommonConfig
    tuning: TuningBudget
    environment: EnvironmentMetadata


@dataclass(frozen=True)
class PairedObservation:
    """One mid-cell feeder swap with its observed order."""

    ordinal: int
    first: SystemRun
    second: SystemRun
    uninterrupted: bool

    def run(self, system: str) -> SystemRun:
        """Return the named side independent of pair order."""
        if self.first.system == system:
            return self.first
        if self.second.system == system:
            return self.second
        raise ValueError(f"paired observation has no {system} run")

    def penalty_percent(self) -> float:
        """Return loader throughput loss relative to the counterfactual."""
        reference = self.run("counterfactual").throughput
        loader = self.run("loader").throughput
        return 100.0 * (reference - loader) / reference

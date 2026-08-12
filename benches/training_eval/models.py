"""Immutable records for live-training paired cells."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DecisionRule:
    """Preregistered collection and interval decision rule."""

    threshold_percent: float
    min_pairs: int = 10
    max_pairs: int = 40
    max_half_width_percent: float = 0.15
    bootstrap_draws: int = 10_000
    bootstrap_seed: int = 0
    mode: str = "upper"


@dataclass(frozen=True)
class TrainingCellConfig:
    """Model, data, and feeder controls shared by both cell halves."""

    evaluation_id: str
    point_id: str
    comparison_kind: str
    subject: str
    reference: str
    workload_family: str
    data_class: str
    batch_size: int
    sequence_length: int
    model_width: int
    model_depth: int
    attention_heads: int
    precision: str
    optimizer: str
    delivery: str
    subject_workers: int
    reference_workers: int
    subject_prefetch: int
    reference_prefetch: int
    half_seconds: float
    tuning_trials: int
    tuning_seconds: float
    tuning_knobs: tuple[str, ...]
    decision: DecisionRule


@dataclass(frozen=True)
class TrainingEnvironment:
    """Machine identity, lease, clocks, and activity controls for one cell."""

    captured_at: str
    machine: str
    operating_system: str
    architecture: str
    python: str
    torch: str
    accelerator: str
    accelerator_clock: str
    memory_clock: str
    cpu_governor: str
    power_profile: str
    plugged_in: bool | None
    thermal_steady: bool
    interactive_load: bool
    commit: str
    lease_kind: str
    lease_token: str
    ambient_probe_id: str

    def stability_key(self) -> tuple[Any, ...]:
        """Return controls that must remain fixed throughout one point decision."""
        return (
            self.machine,
            self.operating_system,
            self.architecture,
            self.python,
            self.torch,
            self.accelerator,
            self.accelerator_clock,
            self.memory_clock,
            self.cpu_governor,
            self.power_profile,
            self.plugged_in,
            self.thermal_steady,
            self.interactive_load,
            self.commit,
            self.lease_kind,
            self.lease_token,
            self.ambient_probe_id,
        )


@dataclass(frozen=True)
class TrainingHalf:
    """One timed feeder half while the training process remains alive."""

    system: str
    process_token: str
    duration_seconds: float
    optimizer_step_start: int
    optimizer_step_stop: int
    samples: int
    rate_steps_per_second: float
    rate_samples_per_second: float
    warmed: bool
    batch_hash_chain: str
    terminal_loss: float | None
    environment: TrainingEnvironment


@dataclass(frozen=True)
class TrainingObservation:
    """One uninterrupted paired feeder swap at one model and data point."""

    ordinal: int
    config: TrainingCellConfig
    first: TrainingHalf
    second: TrainingHalf
    uninterrupted_model_process: bool

    def half(self, system: str) -> TrainingHalf:
        """Return a named half independent of cell order."""
        if self.first.system == system:
            return self.first
        if self.second.system == system:
            return self.second
        raise ValueError(f"training observation has no {system} half")

    def tax_percent(self) -> float:
        """Return subject throughput tax relative to the same-cell reference."""
        reference = self.half(self.config.reference).rate_samples_per_second
        subject = self.half(self.config.subject).rate_samples_per_second
        if os.environ.get("HYPERLOADER_TRAINING_EVAL_MUTATION") == "reverse-tax":
            return 100.0 * (subject - reference) / reference
        return 100.0 * (reference - subject) / reference

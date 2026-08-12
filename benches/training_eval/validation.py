"""Invariants for live-training paired observations."""

from __future__ import annotations

from collections.abc import Sequence

from .models import DecisionRule, TrainingCellConfig, TrainingHalf, TrainingObservation


class TrainingProtocolError(ValueError):
    """A live-training record violates a comparison invariant."""


def validate_observations(observations: Sequence[TrainingObservation]) -> None:
    """Reject incomparable cells, process restarts, and unpinned measurements."""
    if not observations:
        raise TrainingProtocolError("the training point has no observations")
    if [item.ordinal for item in observations] != list(range(len(observations))):
        raise TrainingProtocolError("observation ordinals must be contiguous from zero")
    first = observations[0]
    config = first.config
    _validate_config(config)
    environment_key = first.first.environment.stability_key()
    process_token = first.first.process_token
    for observation in observations:
        if observation.config != config:
            raise TrainingProtocolError("one point decision cannot mix configurations")
        _validate_pair(observation)
        for half in (observation.first, observation.second):
            if half.environment.stability_key() != environment_key:
                raise TrainingProtocolError(
                    "point environment controls changed between cells"
                )
            if half.process_token != process_token:
                raise TrainingProtocolError(
                    "the live training process changed between halves"
                )
    _validate_rule(config.decision)


def _validate_config(config: TrainingCellConfig) -> None:
    positive = (
        config.batch_size,
        config.model_parameters,
        config.dataset_rows,
        config.resident_batches,
        config.warmup_steps,
    )
    if any(value <= 0 for value in positive) or config.learning_rate <= 0:
        raise TrainingProtocolError(
            "training configuration dimensions must be positive"
        )
    required = (
        config.device,
        config.model_name,
        config.optimizer,
        config.delivery,
        config.dataset_identity,
    )
    if not all(required):
        raise TrainingProtocolError("training configuration identity is incomplete")
    if config.data_class == "pretokenized-text":
        token_dimensions = (
            config.sequence_length,
            config.model_width,
            config.model_depth,
            config.attention_heads,
        )
        if config.input_resolution is not None or any(
            value is None or value <= 0 for value in token_dimensions
        ):
            raise TrainingProtocolError("token configuration dimensions are invalid")
    elif config.data_class == "image-folder-standard-augmentation":
        if config.sequence_length is not None or not config.input_resolution:
            raise TrainingProtocolError("image configuration dimensions are invalid")
    if config.comparison_kind != "null" and (
        config.tuning_trials <= 0
        or config.tuning_seconds <= 0
        or not config.tuning_knobs
    ):
        raise TrainingProtocolError(
            "claim-bearing points require counted feeder tuning"
        )


def _validate_pair(observation: TrainingObservation) -> None:
    config = observation.config
    expected_first = (
        config.reference if observation.ordinal % 2 == 0 else config.subject
    )
    if observation.first.system != expected_first:
        raise TrainingProtocolError("pair order must alternate by observation ordinal")
    if {observation.first.system, observation.second.system} != {
        config.reference,
        config.subject,
    }:
        raise TrainingProtocolError(
            "each cell needs the declared subject and reference"
        )
    if not observation.uninterrupted_model_process:
        raise TrainingProtocolError("the model process must remain uninterrupted")
    if observation.second.optimizer_step_start != observation.first.optimizer_step_stop:
        raise TrainingProtocolError(
            "optimizer steps must be contiguous across the swap"
        )
    if observation.first.environment != observation.second.environment:
        raise TrainingProtocolError(
            "both halves require identical environment metadata"
        )
    for half in (observation.first, observation.second):
        _validate_half(half, config.half_seconds)


def _validate_half(half: TrainingHalf, half_seconds: float) -> None:
    if half.duration_seconds < half_seconds or half_seconds <= 0:
        raise TrainingProtocolError(
            "half duration is shorter than the preregistered cell"
        )
    steps = half.optimizer_step_stop - half.optimizer_step_start
    if steps <= 0 or half.samples <= 0:
        raise TrainingProtocolError(
            "each half must complete positive steps and samples"
        )
    if half.rate_steps_per_second <= 0 or half.rate_samples_per_second <= 0:
        raise TrainingProtocolError("training rates must be positive")
    if not half.warmed or not half.batch_hash_chain:
        raise TrainingProtocolError(
            "each half must be warm and carry a batch hash chain"
        )
    environment = half.environment
    if not environment.lease_token or environment.lease_kind not in {
        "SPARK-LOCK",
        "LOCAL-LOCK",
    }:
        raise TrainingProtocolError("a named machine lease is required")
    if environment.interactive_load or not environment.thermal_steady:
        raise TrainingProtocolError(
            "cells require no interactive load and thermal steady state"
        )
    required_pins = (
        environment.accelerator_clock,
        environment.memory_clock,
        environment.cpu_governor,
        environment.power_profile,
        environment.ambient_probe_id,
    )
    if not all(required_pins):
        raise TrainingProtocolError(
            "clock, power, governor, and ambient records are required"
        )
    if environment.lease_kind == "LOCAL-LOCK" and environment.plugged_in is not True:
        raise TrainingProtocolError("local laptop cells require plugged-in power")


def _validate_rule(rule: DecisionRule) -> None:
    if rule.mode not in {"upper", "absolute"}:
        raise TrainingProtocolError("decision mode must be upper or absolute")
    if rule.threshold_percent < 0:
        raise TrainingProtocolError("decision threshold must be nonnegative")
    if not 1 <= rule.min_pairs <= rule.max_pairs:
        raise TrainingProtocolError("pair bounds are invalid")
    if rule.max_half_width_percent <= 0 or rule.bootstrap_draws <= 0:
        raise TrainingProtocolError("precision and bootstrap controls must be positive")

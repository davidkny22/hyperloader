"""Uninterrupted live-training feeder swap behavior tests."""

from __future__ import annotations

import hashlib

import pytest
import torch

from benches.training_eval import (
    DecisionRule,
    ResidentTokenFeeder,
    TokenBatch,
    TrainingCellConfig,
    TrainingEnvironment,
    run_training_observation,
    validate_observations,
    warm_training_process,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class _Runner:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.steps = 0

    def step(self, batch: TokenBatch) -> float:
        self.clock.value += 0.6
        self.steps += 1
        return float(batch.tokens.sum())

    def finish(self, loss: float) -> float:
        return loss


def test_pair_swaps_feeders_without_restarting_optimizer_steps() -> None:
    clock = _Clock()
    runner = _Runner(clock)
    observation = run_training_observation(
        _config(),
        _environment(),
        ordinal=0,
        process_token="one-process",
        optimizer_step_start=10,
        feeders={
            "counterfactual": ResidentTokenFeeder("counterfactual", (_batch(1),)),
            "hyperloader": ResidentTokenFeeder("hyperloader", (_batch(2),)),
        },
        runner=runner,
        warmup_complete=True,
        clock=clock,
    )
    assert observation.first.system == "counterfactual"
    assert observation.second.system == "hyperloader"
    assert observation.first.optimizer_step_start == 10
    assert observation.first.optimizer_step_stop == 12
    assert observation.second.optimizer_step_start == 12
    assert observation.second.optimizer_step_stop == 14
    assert observation.first.process_token == observation.second.process_token
    assert observation.first.duration_seconds == pytest.approx(1.2)
    assert observation.second.duration_seconds == pytest.approx(1.2)
    assert observation.first.batch_hash_chain != observation.second.batch_hash_chain
    validate_observations((observation,))


def test_pair_requires_warmup_and_exact_feeder_names() -> None:
    clock = _Clock()
    with pytest.raises(ValueError, match="warmup"):
        run_training_observation(
            _config(),
            _environment(),
            ordinal=0,
            process_token="one-process",
            optimizer_step_start=0,
            feeders={},
            runner=_Runner(clock),
            warmup_complete=False,
            clock=clock,
        )


def test_warmup_runs_each_feeder_outside_the_timed_pair() -> None:
    clock = _Clock()
    runner = _Runner(clock)
    feeders = {
        "counterfactual": ResidentTokenFeeder("counterfactual", (_batch(1),)),
        "hyperloader": ResidentTokenFeeder("hyperloader", (_batch(2),)),
    }
    steps = warm_training_process(
        feeders,
        runner,
        feeder_order=("counterfactual", "hyperloader"),
        steps_per_feeder=2,
    )
    assert steps == 4
    assert runner.steps == 4


def _batch(value: int) -> TokenBatch:
    return TokenBatch(
        torch.full((2, 4), value, dtype=torch.int64),
        hashlib.sha256(bytes([value])).hexdigest(),
    )


def _config() -> TrainingCellConfig:
    return TrainingCellConfig(
        evaluation_id="eval-live",
        point_id="dial-01",
        comparison_kind="loader-tax",
        subject="hyperloader",
        reference="counterfactual",
        workload_family="dial",
        data_class="pretokenized-text",
        batch_size=2,
        sequence_length=4,
        input_resolution=None,
        model_width=8,
        model_depth=1,
        attention_heads=2,
        precision="float32",
        optimizer="adamw",
        learning_rate=0.0003,
        delivery="host-sync-h2d",
        device="cpu",
        model_name="test transformer",
        model_parameters=1000,
        dataset_rows=64,
        dataset_identity="runtime-dataset",
        seed=0,
        resident_batches=8,
        warmup_steps=3,
        subject_workers=2,
        reference_workers=0,
        subject_prefetch=2,
        reference_prefetch=0,
        half_seconds=1.0,
        tuning_trials=1,
        tuning_seconds=1.0,
        tuning_knobs=("workers",),
        decision=DecisionRule(threshold_percent=1.0, min_pairs=1),
    )


def _environment() -> TrainingEnvironment:
    return TrainingEnvironment(
        captured_at="captured-at-from-record",
        machine="machine-under-test",
        operating_system="operating-system-under-test",
        architecture="architecture-under-test",
        python="runtime-version-from-record",
        torch="provider-version-from-record",
        accelerator="accelerator-under-test",
        accelerator_clock="accelerator-clock-from-record",
        memory_clock="memory-clock-from-record",
        cpu_governor="governor-from-record",
        power_profile="power-profile-from-record",
        plugged_in=None,
        thermal_steady=True,
        interactive_load=False,
        commit="source-revision-from-record",
        lease_kind="SPARK-LOCK",
        lease_token="deadbeef",
        ambient_probe_id="ambient-1",
    )

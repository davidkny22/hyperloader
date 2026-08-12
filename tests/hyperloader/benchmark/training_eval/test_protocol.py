"""Live-training protocol behavior tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from benches.training_eval import (
    DecisionRule,
    TrainingCellConfig,
    TrainingEnvironment,
    TrainingHalf,
    TrainingObservation,
    TrainingProtocolError,
    decide,
    validate_observations,
)


def _environment() -> TrainingEnvironment:
    return TrainingEnvironment(
        captured_at="2026-08-12T20:00:00+00:00",
        machine="spark",
        operating_system="Linux",
        architecture="aarch64",
        python="3.12.3",
        torch="2.13.0",
        accelerator="GB10",
        accelerator_clock="2400 MHz",
        memory_clock="reported",
        cpu_governor="performance",
        power_profile="MAXN",
        plugged_in=None,
        thermal_steady=True,
        interactive_load=False,
        commit="0123456",
        lease_kind="SPARK-LOCK",
        lease_token="deadbeef",
        ambient_probe_id="ambient-1",
    )


def _config(*, mode: str = "upper") -> TrainingCellConfig:
    return TrainingCellConfig(
        evaluation_id="eval-1",
        point_id="dial-01",
        comparison_kind="loader-tax",
        subject="hyperloader",
        reference="counterfactual",
        workload_family="dial",
        data_class="pretokenized-text",
        batch_size=8,
        sequence_length=512,
        model_width=256,
        model_depth=4,
        attention_heads=4,
        precision="bf16",
        optimizer="adamw",
        delivery="host-sync-h2d",
        subject_workers=4,
        reference_workers=0,
        subject_prefetch=4,
        reference_prefetch=0,
        half_seconds=45.0,
        tuning_trials=6,
        tuning_seconds=12.0,
        tuning_knobs=("workers", "prefetch_factor"),
        decision=DecisionRule(threshold_percent=1.0, mode=mode),
    )


def _observations(count: int, tax: float = 0.5) -> list[TrainingObservation]:
    environment = _environment()
    config = _config()
    observations = []
    for ordinal in range(count):
        reference_rate = 1000.0
        subject_rate = reference_rate * (1.0 - tax / 100.0)
        start = ordinal * 200
        reference = TrainingHalf(
            "counterfactual",
            "process-a",
            45.0,
            start,
            start + 100,
            800,
            100 / 45.0,
            reference_rate,
            True,
            f"ref-{ordinal}",
            1.0,
            environment,
        )
        subject = replace(
            reference,
            system="hyperloader",
            optimizer_step_start=start + 100,
            optimizer_step_stop=start + 200,
            rate_samples_per_second=subject_rate,
            batch_hash_chain=f"subject-{ordinal}",
        )
        first, second = (reference, subject) if ordinal % 2 == 0 else (subject, reference)
        if ordinal % 2:
            first = replace(first, optimizer_step_start=start, optimizer_step_stop=start + 100)
            second = replace(second, optimizer_step_start=start + 100, optimizer_step_stop=start + 200)
        observations.append(TrainingObservation(ordinal, config, first, second, True))
    return observations


def test_decision_collects_then_passes_on_upper_interval() -> None:
    assert decide(_observations(9)).status == "collect"
    result = decide(_observations(10))
    assert result.status == "pass"
    assert result.mean_tax_percent == pytest.approx(0.5)


def test_process_restart_and_noncontiguous_swap_are_rejected() -> None:
    observations = _observations(10)
    observations[0] = replace(
        observations[0],
        second=replace(observations[0].second, process_token="process-b"),
    )
    with pytest.raises(TrainingProtocolError, match="process changed"):
        validate_observations(observations)

    observations = _observations(10)
    observations[0] = replace(
        observations[0],
        second=replace(observations[0].second, optimizer_step_start=999),
    )
    with pytest.raises(TrainingProtocolError, match="contiguous"):
        validate_observations(observations)


def test_interactive_or_unleased_cell_is_rejected() -> None:
    observations = _observations(10)
    changed = replace(observations[0].first.environment, interactive_load=True)
    observations[0] = replace(
        observations[0],
        first=replace(observations[0].first, environment=changed),
        second=replace(observations[0].second, environment=changed),
    )
    with pytest.raises(TrainingProtocolError, match="interactive load"):
        validate_observations(observations)


def test_null_mode_uses_absolute_interval_bound() -> None:
    observations = _observations(10, tax=-0.2)
    config = replace(
        observations[0].config,
        comparison_kind="null",
        subject="null-b",
        reference="null-a",
        decision=DecisionRule(threshold_percent=0.5, mode="absolute"),
    )
    converted = []
    for observation in observations:
        halves = {
            "counterfactual": replace(observation.half("counterfactual"), system="null-a"),
            "hyperloader": replace(observation.half("hyperloader"), system="null-b"),
        }
        first_system = "null-a" if observation.ordinal % 2 == 0 else "null-b"
        second_system = "null-b" if observation.ordinal % 2 == 0 else "null-a"
        converted.append(
            TrainingObservation(
                observation.ordinal,
                config,
                halves["counterfactual" if first_system == "null-a" else "hyperloader"],
                halves["counterfactual" if second_system == "null-a" else "hyperloader"],
                True,
            )
        )
    assert decide(converted).status == "pass"

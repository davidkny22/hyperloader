"""End-to-end behavior for token-point collection."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from benches.training_eval.dial import TransformerDialPoint
from benches.training_eval.models import (
    DecisionRule,
    TrainingCellConfig,
    TrainingEnvironment,
)
from benches.training_eval.token_point import collect_token_point
from benches.training_eval.token_source import PretokenizedRows
from benches.training_eval.transformer import DialTransformer


def test_pretokenized_rows_are_seeded_and_finite() -> None:
    first = PretokenizedRows(rows=4, sequence_length=5, vocabulary_size=17, seed=9)
    second = PretokenizedRows(rows=4, sequence_length=5, vocabulary_size=17, seed=9)
    changed = PretokenizedRows(rows=4, sequence_length=5, vocabulary_size=17, seed=10)
    assert len(first) == 4
    assert torch.equal(first[2], second[2])
    assert not torch.equal(first[2], changed[2])


def test_null_point_collects_incremental_evidence_and_decision(tmp_path: Path) -> None:
    observations = tmp_path / "observations.jsonl"
    decision = tmp_path / "decision.json"
    point = TransformerDialPoint("tiny", 8, 1, 2, 4, 2, 17)
    result = collect_token_point(
        _config(),
        _environment(),
        model_factory=lambda: DialTransformer(point),
        device=torch.device("cpu"),
        seed=3,
        bank_batches=2,
        warmup_steps=1,
        pin_memory=False,
        observations_path=observations,
        decision_path=decision,
    )
    rows = observations.read_text(encoding="utf-8").splitlines()
    document = json.loads(decision.read_text(encoding="utf-8"))
    assert result.status == "pass"
    assert len(rows) == 1
    assert json.loads(rows[0])["uninterrupted_model_process"] is True
    assert document["decision"]["pairs"] == 1
    assert document["observations"] == observations.name


def test_token_point_refuses_to_replace_evidence(tmp_path: Path) -> None:
    observations = tmp_path / "observations.jsonl"
    observations.write_text("existing\n", encoding="utf-8")
    point = TransformerDialPoint("tiny", 8, 1, 2, 4, 2, 17)
    try:
        collect_token_point(
            _config(),
            _environment(),
            model_factory=lambda: DialTransformer(point),
            device=torch.device("cpu"),
            seed=3,
            bank_batches=2,
            warmup_steps=1,
            pin_memory=False,
            observations_path=observations,
            decision_path=tmp_path / "decision.json",
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing evidence was not rejected")


def _config() -> TrainingCellConfig:
    return TrainingCellConfig(
        evaluation_id="test-evaluation",
        point_id="null",
        comparison_kind="null",
        subject="null-b",
        reference="null-a",
        workload_family="transformer-dial",
        data_class="pretokenized-text",
        batch_size=2,
        sequence_length=4,
        model_width=8,
        model_depth=1,
        attention_heads=2,
        precision="float32",
        optimizer="AdamW(lr=0.0003)",
        delivery="pageable",
        subject_workers=0,
        reference_workers=0,
        subject_prefetch=1,
        reference_prefetch=1,
        half_seconds=0.001,
        tuning_trials=0,
        tuning_seconds=0.0,
        tuning_knobs=("workers", "prefetch"),
        decision=DecisionRule(
            threshold_percent=100.0,
            min_pairs=1,
            max_pairs=1,
            max_half_width_percent=100.0,
            bootstrap_draws=100,
            mode="absolute",
        ),
    )


def _environment() -> TrainingEnvironment:
    return TrainingEnvironment(
        captured_at="runtime-captured",
        machine="runtime-machine",
        operating_system="runtime-system",
        architecture="runtime-architecture",
        python="runtime-python",
        torch="runtime-torch",
        accelerator="runtime-accelerator",
        accelerator_clock="runtime-clock",
        memory_clock="runtime-memory-clock",
        cpu_governor="runtime-governor",
        power_profile="runtime-profile",
        plugged_in=None,
        thermal_steady=True,
        interactive_load=False,
        commit="runtime-commit",
        lease_kind="SPARK-LOCK",
        lease_token="deadbeef",
        ambient_probe_id="runtime-probe",
    )

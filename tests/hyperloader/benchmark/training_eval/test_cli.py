"""Machine-readable training-evaluation command behavior."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from benches.training_eval import (
    DecisionRule,
    TrainingCellConfig,
    TrainingEnvironment,
    TrainingHalf,
    TrainingObservation,
)

ROOT = Path(__file__).parents[4]


def test_module_entry_decodes_cells_and_writes_the_decision(tmp_path: Path) -> None:
    observations = tmp_path / "observations.jsonl"
    output = tmp_path / "decision.json"
    observations.write_text(
        json.dumps(asdict(_observation())) + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benches.training_evaluation",
            "--observations",
            str(observations),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["schema_version"] == 1
    assert result["kind"] == "training-throughput-decision"
    assert result["decision"]["status"] == "pass"
    assert result["decision"]["pairs"] == 1


def _observation() -> TrainingObservation:
    environment = TrainingEnvironment(
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
        ambient_probe_id="ambient-from-record",
    )
    config = TrainingCellConfig(
        evaluation_id="evaluation-under-test",
        point_id="point-under-test",
        comparison_kind="loader-tax",
        subject="hyperloader",
        reference="counterfactual",
        workload_family="dial",
        data_class="pretokenized-text",
        batch_size=2,
        sequence_length=4,
        model_width=8,
        model_depth=1,
        attention_heads=2,
        precision="float32",
        optimizer="adamw",
        delivery="host-sync-h2d",
        subject_workers=1,
        reference_workers=0,
        subject_prefetch=1,
        reference_prefetch=0,
        half_seconds=1.0,
        tuning_trials=1,
        tuning_seconds=1.0,
        tuning_knobs=("workers",),
        decision=DecisionRule(
            threshold_percent=1.0,
            min_pairs=1,
            max_pairs=1,
            bootstrap_draws=100,
        ),
    )
    reference = TrainingHalf(
        "counterfactual",
        "process-under-test",
        1.0,
        0,
        1,
        2,
        1.0,
        100.0,
        True,
        "reference-chain",
        1.0,
        environment,
    )
    subject = TrainingHalf(
        "hyperloader",
        "process-under-test",
        1.0,
        1,
        2,
        2,
        1.0,
        99.5,
        True,
        "subject-chain",
        1.0,
        environment,
    )
    return TrainingObservation(0, config, reference, subject, True)

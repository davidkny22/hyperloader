"""Benchmark-registry projection from verified training points."""

from __future__ import annotations

from copy import deepcopy

import pytest

from benches.training_eval.registry_projection import project_registry_records


def _bundle() -> dict[str, object]:
    config = {
        "point_id": "dial-01",
        "subject": "hyperloader",
        "decision": {"max_half_width_percent": 0.15},
        "batch_size": 16,
    }
    environment = {
        "captured_at": "2026-08-13T03:00:00+00:00",
        "machine": "machine from run",
        "operating_system": "Linux",
        "architecture": "runtime architecture",
        "python": "runtime Python",
        "torch": "runtime Torch",
        "accelerator": "runtime accelerator",
        "accelerator_clock": "runtime clock",
        "commit": "abcdef0123456789",
        "machine_state_control": "native-alu-pulse",
        "machine_state_cpus": [2, 3],
        "machine_state_active_microseconds": 4,
        "machine_state_period_microseconds": 400,
    }
    decision = {
        "status": "pass",
        "pairs": 40,
        "mean_tax_percent": 1.5,
        "lower_percent": 1.0,
        "upper_percent": 2.0,
        "half_width_percent": 0.5,
    }
    return {
        "commit": "abcdef0123456789",
        "points": [
            {
                "point": "remote-root/dial-01-hyperloader",
                "config": config,
                "environment": environment,
                "decision": decision,
            }
        ],
    }


def test_projection_carries_measured_interval_controls_and_evidence() -> None:
    records = project_registry_records(_bundle(), evidence_root="results/campaign")

    assert len(records) == 1
    record = records[0]
    assert record["id"] == "training-eval-machine-from-run-dial-01-hyperloader-abcdef0"
    assert record["interval"]["terminal_reason"] == "max-pair cap"
    assert record["config"]["machine_state_cpus"] == [2, 3]
    assert record["config"]["evidence"] == "results/campaign/dial-01-hyperloader"
    assert record["status"] == "verified"


def test_projection_rejects_commit_drift_and_duplicate_identities() -> None:
    bundle = _bundle()
    changed = deepcopy(bundle)
    changed["points"][0]["environment"]["commit"] = "another"
    with pytest.raises(ValueError, match="commit-matched"):
        project_registry_records(changed, evidence_root="results/campaign")

    duplicate = deepcopy(bundle)
    duplicate["points"].append(deepcopy(duplicate["points"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        project_registry_records(duplicate, evidence_root="results/campaign")


def test_projection_marks_precision_terminal_when_half_width_reaches_target() -> None:
    bundle = _bundle()
    bundle["points"][0]["decision"]["half_width_percent"] = 0.1

    [record] = project_registry_records(bundle, evidence_root="results/campaign")

    assert record["interval"]["terminal_reason"] == "precision target"

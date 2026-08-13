"""Behavioral reconciliation of completed training campaigns."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from benches.training_eval.decision import decide
from benches.training_eval.result_bundle import CampaignEvidenceError, build_bundle

from .test_protocol import _observations


def _campaign(root: Path, *, commit: str = "commit-from-run") -> Path:
    point = root / "dial-01-hyperloader"
    point.mkdir(parents=True)
    observations = _observations(10)
    environment = replace(
        observations[0].first.environment,
        commit=commit,
        machine_state_control="native-alu-pulse",
        machine_state_cpus=(2, 3),
        machine_state_active_microseconds=4,
        machine_state_period_microseconds=400,
    )
    observations = [
        replace(
            item,
            first=replace(item.first, environment=environment),
            second=replace(item.second, environment=environment),
        )
        for item in observations
    ]
    (point / "observations.jsonl").write_text(
        "".join(
            json.dumps(asdict(item), sort_keys=True) + "\n" for item in observations
        ),
        encoding="utf-8",
    )
    (point / "decision.json").write_text(
        json.dumps({"decision": asdict(decide(observations))}), encoding="utf-8"
    )
    (point / "machine-state.json").write_text(
        json.dumps(
            {
                "command_returncode": 0,
                "control": "native-alu-pulse",
                "cores": [2, 3],
                "active_microseconds": 4,
                "period_microseconds": 400,
            }
        ),
        encoding="utf-8",
    )
    (root / "dial-01-hyperloader-clock.json").write_text(
        json.dumps({"command_returncode": 0, "reset_stdout": "reset"}),
        encoding="utf-8",
    )
    (root / "campaign.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "completed": [{"output": "remote-root/dial-01-hyperloader"}],
            }
        ),
        encoding="utf-8",
    )
    return point


def test_bundle_recomputes_terminal_decision_and_traces_both_guards(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    _campaign(root)

    bundle = build_bundle([root], expected_commit="commit-from-run")

    assert bundle["commit"] == "commit-from-run"
    assert len(bundle["points"]) == 1
    point = bundle["points"][0]
    assert point["observations"] == 10
    assert point["decision"]["status"] == "pass"
    assert point["environment"]["machine_state_cpus"] == (2, 3)


def test_bundle_rejects_commit_drift_and_stale_decision(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    point = _campaign(root)
    with pytest.raises(CampaignEvidenceError, match="commit"):
        build_bundle([root], expected_commit="another-commit")

    decision_path = point / "decision.json"
    document = json.loads(decision_path.read_text(encoding="utf-8"))
    document["decision"]["mean_tax_percent"] = 99.0
    decision_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CampaignEvidenceError, match="decision"):
        build_bundle([root], expected_commit="commit-from-run")


def test_bundle_rejects_missing_guard_cleanup(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    _campaign(root)
    (root / "dial-01-hyperloader-clock.json").write_text(
        json.dumps({"command_returncode": 0, "reset_stdout": ""}), encoding="utf-8"
    )

    with pytest.raises(CampaignEvidenceError, match="clock guard"):
        build_bundle([root], expected_commit="commit-from-run")

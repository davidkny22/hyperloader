"""Local dominance lease and steady-ambient behavior tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benches.benchmark_protocol.matrix import workload_names
from benches.training_eval import AmbientProbe, TrainingDecision
from benches.training_eval.local_dominance import run_local_dominance


def test_local_campaign_runs_all_six_cells_under_owned_lock(tmp_path: Path) -> None:
    observed: dict[str, Any] = {}

    def campaign(**values: Any) -> dict[str, Any]:
        observed.update(values)
        values["output"].mkdir()
        return {"status": "smoke"}

    lock = tmp_path / "LOCAL-LOCK"
    output = tmp_path / "run"
    source_revision = f"revision-{tmp_path.name}"
    lease_claimant = f"claimant-{tmp_path.name}"
    summary = run_local_dominance(
        output=output,
        lock_path=lock,
        prior_ambient=_probe("prior", 100.0),
        current_ambient=_probe("current", 100.1),
        null_decision=_null_decision(0.2),
        commit=source_revision,
        lease_claimant=lease_claimant,
        cpu_governor="profile-from-run",
        gpu_clock="clock-from-run",
        worker_cpus=(2, 4),
        torchvision_version="provider-version-from-run",
        smoke=True,
        campaign=campaign,
    )
    assert summary == {"status": "smoke"}
    assert observed["workloads"] == workload_names()
    assert observed["references"] == ("torch", "spdl")
    assert observed["environment"].concurrent_load is False
    assert observed["environment"].commit == source_revision
    assert observed["environment"].machine
    assert observed["worker_cpus"] == (2, 4)
    assert not lock.exists()
    session = json.loads((output / "local-session.json").read_text())
    assert session["ambient"]["decision"]["status"] == "pass"
    assert session["lease"]["claimant"] == lease_claimant


def test_local_campaign_rejects_ambient_drift_before_claiming(tmp_path: Path) -> None:
    called = False

    def campaign(**_: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    lock = tmp_path / "LOCAL-LOCK"
    with pytest.raises(RuntimeError, match="ambient"):
        run_local_dominance(
            output=tmp_path / "run",
            lock_path=lock,
            prior_ambient=_probe("prior", 100.0),
            current_ambient=_probe("current", 101.0),
            null_decision=_null_decision(0.2),
            commit=f"revision-{tmp_path.name}",
            lease_claimant=f"claimant-{tmp_path.name}",
            cpu_governor="profile-from-run",
            gpu_clock="clock-from-run",
            worker_cpus=(2, 4),
            torchvision_version="provider-version-from-run",
            smoke=True,
            campaign=campaign,
        )
    assert called is False
    assert not lock.exists()


def _probe(probe_id: str, rate: float) -> AmbientProbe:
    return AmbientProbe(probe_id, rate, 2.0, 10.0, 1.0, 1_000_000)


def _null_decision(bound: float) -> TrainingDecision:
    return TrainingDecision(
        "pass", 10, 0.0, -bound, bound, bound, bound + 0.1, "absolute"
    )

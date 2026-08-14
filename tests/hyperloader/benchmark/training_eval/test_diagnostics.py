"""Behavioral tests for bounded training residual diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest

from benches.training_eval.diagnostics.analyze import analyze_gil_profile
from benches.training_eval.diagnostics.cpu_activity import diff_cpu_activity
from benches.training_eval.diagnostics.segments import summarize_timings


def test_cpu_activity_reports_each_runtime_core_without_topology_pins() -> None:
    rows = diff_cpu_activity(
        {"cpu2": (10, 0, 0, 90, 0), "cpu7": (5, 0, 0, 95, 0)},
        {"cpu2": (30, 0, 0, 170, 0), "cpu7": (5, 0, 0, 195, 0)},
    )
    assert [row["cpu"] for row in rows] == ["cpu2", "cpu7"]
    assert rows[0]["utilization_percent"] == pytest.approx(20.0)
    assert rows[1]["utilization_percent"] == pytest.approx(0.0)


def test_segment_summary_preserves_host_and_cuda_fields() -> None:
    summary = summarize_timings(
        [
            {"cuda_copy_ms": 1.0, "host_sync_ms": 4.0},
            {"cuda_copy_ms": 3.0, "host_sync_ms": 8.0},
        ]
    )
    assert summary["cuda_copy_ms"]["mean"] == pytest.approx(2.0)
    assert summary["host_sync_ms"]["median"] == pytest.approx(6.0)
    assert summary["host_sync_ms"]["p95"] == pytest.approx(7.8)


def test_gil_analysis_attributes_system_stage_and_sample_rate(tmp_path: Path) -> None:
    raw = tmp_path / "profile.raw"
    raw.write_text(
        "thread;profile_hyperloader_half;profile_next_batch;leaf 4\n"
        "thread;profile_hyperloader_half;profile_sync;leaf 2\n"
        "thread;profile_counterfactual_half;profile_compute;leaf 3\n",
        encoding="utf-8",
    )
    report = analyze_gil_profile(
        raw, total_seconds_by_system={"hyperloader": 2.0, "counterfactual": 3.0}
    )
    hyperloader = report["systems"]["hyperloader"]
    assert hyperloader["gil_samples"] == 6
    assert hyperloader["gil_samples_per_second"] == pytest.approx(3.0)
    assert hyperloader["stage_samples"][0]["name"] == "next_batch"
    assert report["systems"]["counterfactual"]["gil_samples_per_second"] == 1.0

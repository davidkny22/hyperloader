"""Ambient, lease, and output boundary behavior tests."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from benches.training_eval import (
    AmbientProbe,
    FileLease,
    LeaseRecord,
    LeaseUnavailable,
    compare_ambient,
    write_result,
)


def test_ambient_probe_accepts_inside_band_and_rejects_outside() -> None:
    prior = AmbientProbe("a", 100.0, 10.0, 20.0, 2.0, 1_000)
    inside = AmbientProbe("b", 100.4, 10.0, 21.0, 2.5, 1_100)
    outside = AmbientProbe("c", 102.0, 10.0, 21.0, 2.5, 1_100)
    assert compare_ambient(prior, inside, null_band_percent=0.5).status == "pass"
    assert compare_ambient(prior, outside, null_band_percent=0.5).status == "fail"


def test_file_lease_excludes_live_claim_and_releases_owned_token(tmp_path: Path) -> None:
    path = tmp_path / "LOCAL-LOCK"
    lease = FileLease.claim(
        path,
        task_row="T087",
        purpose="local training cell",
        verify_delay_seconds=0,
    )
    with pytest.raises(LeaseUnavailable, match="belongs"):
        FileLease.claim(
            path,
            task_row="T087",
            purpose="contender",
            verify_delay_seconds=0,
        )
    lease.release()
    assert not path.exists()


def test_stale_lease_requires_negative_process_probe(tmp_path: Path) -> None:
    path = tmp_path / "LOCAL-LOCK"
    now = datetime(2026, 8, 12, 20, tzinfo=timezone.utc)
    stale = LeaseRecord(now - timedelta(minutes=61), "T087", "1234abcd", "old")
    path.write_text(stale.render(), encoding="utf-8")
    with pytest.raises(LeaseUnavailable):
        FileLease.claim(
            path,
            task_row="T087",
            purpose="new",
            now=now,
            active_process=lambda: True,
            verify_delay_seconds=0,
        )
    lease = FileLease.claim(
        path,
        task_row="T087",
        purpose="new",
        now=now,
        active_process=lambda: False,
        verify_delay_seconds=0,
    )
    assert lease.record.token != stale.token
    lease.release()


def test_output_is_canonical_json_and_visual_targets_are_rejected(tmp_path: Path) -> None:
    output = tmp_path / "point.json"
    write_result(output, {"kind": "training-throughput-decision", "value": 1})
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1
    with pytest.raises(ValueError, match="machine-readable"):
        write_result(tmp_path / "curve.png", {"value": 1})

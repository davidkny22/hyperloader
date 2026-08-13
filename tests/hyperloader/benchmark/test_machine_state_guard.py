"""Lifecycle behavior for the Spark training machine-state guard."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from benches.spark_machine_state_guard import run_guard


class _RecordingSpinner:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def start(self) -> None:
        self._events.append("start")

    def stop(self) -> dict[str, object]:
        self._events.append("stop")
        return {"threads": [{"core": 2}, {"core": 3}]}


def test_guard_surrounds_the_child_and_records_the_runtime_actuator() -> None:
    events: list[str] = []

    def factory(*args, **kwargs):
        assert args[1] == (2, 3)
        assert kwargs == {"active_microseconds": 4, "period_microseconds": 400}
        return _RecordingSpinner(events)

    def run(command, *, check):
        assert check is True
        events.append(f"run:{command[0]}")
        return subprocess.CompletedProcess(command, 0)

    with TemporaryDirectory() as directory:
        evidence = Path(directory) / "control.json"
        with patch("benches.spark_machine_state_guard.subprocess.run", side_effect=run):
            run_guard(
                evidence=evidence,
                spinner_library=Path("runtime-spinner.so"),
                cores=(2, 3),
                active_microseconds=4,
                period_microseconds=400,
                command=["training-command"],
                spinner_factory=factory,
            )
        record = json.loads(evidence.read_text(encoding="utf-8"))

    assert events == ["start", "run:training-command", "stop"]
    assert record["control"] == "native-alu-pulse"
    assert record["command_returncode"] == 0
    assert record["actuator"]["threads"] == [{"core": 2}, {"core": 3}]


def test_guard_stops_the_actuator_when_the_child_fails() -> None:
    events: list[str] = []

    def factory(*args, **kwargs):
        return _RecordingSpinner(events)

    failure = subprocess.CalledProcessError(7, ["training-command"])
    with TemporaryDirectory() as directory:
        evidence = Path(directory) / "control.json"
        with (
            patch(
                "benches.spark_machine_state_guard.subprocess.run",
                side_effect=failure,
            ),
            pytest.raises(subprocess.CalledProcessError),
        ):
            run_guard(
                evidence=evidence,
                spinner_library=Path("runtime-spinner.so"),
                cores=(2,),
                active_microseconds=4,
                period_microseconds=400,
                command=["training-command"],
                spinner_factory=factory,
            )
        record = json.loads(evidence.read_text(encoding="utf-8"))

    assert events == ["start", "stop"]
    assert record["command_returncode"] == 7

"""Sequential behavior for guarded Spark dial campaigns."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from benches.training_eval.spark_dial_campaign import (
    DialCell,
    build_command,
    parse_cell,
    run_campaign,
)


def _arguments(output_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        cells=[DialCell(1, "torch"), DialCell(1, "hyperloader")],
        output_root=output_root,
        evaluation_id="evaluation-from-run",
        machine="machine-from-run",
        commit="commit-from-run",
        lease_token="lease-from-run",
        clock_mhz=1234,
        accelerator_clock="clock-from-run",
        memory_clock="memory-clock-from-run",
        cpu_governor="governor-from-run",
        power_profile="power-from-run",
        ambient_probe_id="ambient-from-run",
        cpu_set="available-cpus-from-run",
        pythonpath="dependencies-from-run",
        spinner_library=Path("spinner-from-run.so"),
        machine_state_cpu=[2, 3],
        machine_state_active_us=4,
        machine_state_period_us=400,
        half_seconds=1.0,
        min_pairs=2,
        max_pairs=3,
        max_half_width_percent=0.5,
        threshold_percent=5.0,
        bank_batches=8,
        warmup_steps=1,
        bootstrap_draws=100,
        tuning_candidate=["1:1", "2:2"],
        tuning_seconds=0.1,
        tuning_warmup_steps=1,
    )


def test_campaign_runs_cells_serially_and_updates_machine_readable_progress() -> None:
    with TemporaryDirectory() as directory:
        arguments = _arguments(Path(directory) / "campaign")
        commands: list[list[str]] = []

        def run(command: list[str], *, check: bool):
            assert check is True
            commands.append(command)
            return subprocess.CompletedProcess(command, 0)

        with patch(
            "benches.training_eval.spark_dial_campaign.subprocess.run",
            side_effect=run,
        ):
            result = run_campaign(arguments)
        saved = json.loads(
            (arguments.output_root / "campaign.json").read_text(encoding="utf-8")
        )

    assert [command[command.index("--subject") + 1] for command in commands] == [
        "torch",
        "hyperloader",
    ]
    assert result["status"] == saved["status"] == "complete"
    assert len(saved["completed"]) == 2


def test_command_applies_identical_runtime_machine_state_to_both_guards() -> None:
    with TemporaryDirectory() as directory:
        arguments = _arguments(Path(directory))
        command = build_command(arguments, DialCell(4, "torch"), Path(directory) / "p")

    assert command[command.index("--clock-mhz") + 1] == "1234"
    assert command.count("--core") == 2
    assert command.count("--machine-state-cpu") == 2
    assert "available-cpus-from-run" in command
    assert "PYTHONPATH=dependencies-from-run" in command
    assert command[command.index("--dial-index") + 1] == "4"


@pytest.mark.parametrize("value", ["", "x:torch", "9:torch", "1:spdl"])
def test_invalid_campaign_cell_is_rejected(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_cell(value)

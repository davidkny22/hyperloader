"""Sequential behavior for guarded Spark named-anchor campaigns."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from benches.training_eval.spark_anchor_campaign import (
    AnchorCell,
    build_command,
    parse_cell,
    run_campaign,
)

from .test_spark_dial_campaign import _arguments as dial_arguments


def _arguments(output_root: Path) -> argparse.Namespace:
    arguments = dial_arguments(output_root)
    arguments.cells = [AnchorCell("gpt2-124m", "spdl"), AnchorCell("vision", "torch")]
    arguments.image_root = Path("image-root-from-run")
    arguments.resolution = 224
    arguments.vision_batch_size = 32
    return arguments


def test_campaign_runs_named_anchors_and_dials_through_their_public_points() -> None:
    with TemporaryDirectory() as directory:
        arguments = _arguments(Path(directory) / "campaign")
        arguments.cells.append(AnchorCell("dial-7", "hyperloader"))
        commands: list[list[str]] = []

        def run(command: list[str], *, check: bool):
            assert check is True
            commands.append(command)
            return subprocess.CompletedProcess(command, 0)

        with patch(
            "benches.training_eval.spark_anchor_campaign.subprocess.run",
            side_effect=run,
        ):
            result = run_campaign(arguments)

    assert result["status"] == "complete"
    assert "benches.spark_training_point" in commands[0]
    assert "gpt2-124m" in commands[0]
    assert "benches.spark_vision_point" in commands[1]
    assert "image-root-from-run" in commands[1]
    assert "benches.spark_training_point" in commands[2]
    assert commands[2][commands[2].index("--dial-index") + 1] == "7"
    assert result["kind"] == "spark-training-scoped-campaign"


def test_anchor_commands_share_the_runtime_machine_state_builder() -> None:
    with TemporaryDirectory() as directory:
        arguments = _arguments(Path(directory))
        command = build_command(
            arguments, AnchorCell("vision", "hyperloader"), Path(directory) / "v"
        )

    assert command.count("--core") == 2
    assert command.count("--machine-state-cpu") == 2
    assert command[command.index("--subject") + 1] == "hyperloader"
    assert command[command.index("--batch-size") + 1] == "32"


def test_vision_anchor_requires_a_runtime_image_root() -> None:
    with TemporaryDirectory() as directory:
        arguments = _arguments(Path(directory))
        arguments.image_root = None
        with pytest.raises(ValueError, match="image root"):
            run_campaign(arguments)


@pytest.mark.parametrize("value", ["", "dial:torch", "vision:none", "gpt:spdl"])
def test_invalid_anchor_cell_is_rejected(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_cell(value)


def test_runtime_dial_cell_is_parsed_without_a_fixed_campaign_index() -> None:
    assert parse_cell("dial-1:torch") == AnchorCell("dial-1", "torch")
    assert parse_cell("dial-8:hyperloader") == AnchorCell("dial-8", "hyperloader")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_cell("dial-9:hyperloader")

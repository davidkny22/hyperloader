"""Sequential guarded Spark execution for transformer dial points."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .spark_runtime import add_spark_runtime_arguments, guarded_point_command


@dataclass(frozen=True)
class DialCell:
    """One loader and dial-index pair in a Spark campaign."""

    dial_index: int
    subject: str


def parse_cell(value: str) -> DialCell:
    """Parse a caller-supplied dial-index and loader pair."""
    index_text, separator, subject = value.partition(":")
    if not separator or subject not in {"torch", "hyperloader"}:
        raise argparse.ArgumentTypeError("cells use DIAL_INDEX:torch|hyperloader")
    try:
        index = int(index_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("dial index must be an integer") from error
    if index not in range(1, 9):
        raise argparse.ArgumentTypeError("dial index must be between one and eight")
    return DialCell(index, subject)


def run_campaign(arguments: argparse.Namespace) -> dict[str, object]:
    """Run requested dial cells serially and return their completed identities."""
    if not arguments.cells or len(set(arguments.cells)) != len(arguments.cells):
        raise ValueError("campaign cells must be a nonempty unique sequence")
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, object]] = []
    for cell in arguments.cells:
        output = arguments.output_root / f"dial-{cell.dial_index:02d}-{cell.subject}"
        if output.exists():
            raise FileExistsError(f"campaign output already exists: {output}")
        command = build_command(arguments, cell, output)
        subprocess.run(command, check=True)
        completed.append(
            {
                "dial_index": cell.dial_index,
                "subject": cell.subject,
                "output": str(output),
                "decision": str(output / "decision.json"),
                "machine_state": str(output / "machine-state.json"),
                "clock": str(output.with_name(output.name + "-clock.json")),
            }
        )
        _write_summary(arguments.output_root, completed, status="running")
    return _write_summary(arguments.output_root, completed, status="complete")


def build_command(
    arguments: argparse.Namespace, cell: DialCell, output: Path
) -> list[str]:
    """Build one fully guarded public-path point command from runtime facts."""
    return guarded_point_command(
        arguments,
        output=output,
        module="benches.spark_training_point",
        point_arguments=(
            "--kind",
            "dial",
            "--dial-index",
            str(cell.dial_index),
            "--subject",
            cell.subject,
        ),
    )


def _write_summary(
    output_root: Path, completed: list[dict[str, object]], *, status: str
) -> dict[str, object]:
    record = {
        "kind": "spark-training-dial-campaign",
        "status": status,
        "captured_at": datetime.now(UTC).isoformat(),
        "completed": completed,
    }
    path = output_root / "campaign.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return record


def parser() -> argparse.ArgumentParser:
    """Return the runtime-only Spark dial campaign parser."""
    result = argparse.ArgumentParser()
    result.add_argument("--cell", dest="cells", type=parse_cell, action="append")
    result.add_argument("--output-root", type=Path, required=True)
    add_spark_runtime_arguments(result)
    return result


def main() -> None:
    """Run a runtime-defined sequence of guarded Spark dial cells."""
    print(run_campaign(parser().parse_args()))


if __name__ == "__main__":
    main()

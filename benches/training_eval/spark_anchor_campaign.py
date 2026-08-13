"""Sequential guarded Spark execution for named training anchors."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .spark_runtime import add_spark_runtime_arguments, guarded_point_command


@dataclass(frozen=True)
class AnchorCell:
    """One named workload and loader pair in a Spark campaign."""

    kind: str
    subject: str


def parse_cell(value: str) -> AnchorCell:
    """Parse a named workload and loader pair."""
    kind, separator, subject = value.partition(":")
    if (
        not separator
        or kind not in {"gpt2-124m", "gpt2-355m", "vision"}
        or subject not in {"torch", "hyperloader", "spdl"}
    ):
        raise argparse.ArgumentTypeError(
            "anchor cells use gpt2-124m|gpt2-355m|vision:torch|hyperloader|spdl"
        )
    return AnchorCell(kind, subject)


def run_campaign(arguments: argparse.Namespace) -> dict[str, object]:
    """Run requested anchor cells serially and record completed identities."""
    if not arguments.cells or len(set(arguments.cells)) != len(arguments.cells):
        raise ValueError("anchor cells must be a nonempty unique sequence")
    if (
        any(cell.kind == "vision" for cell in arguments.cells)
        and not arguments.image_root
    ):
        raise ValueError("vision anchors require an image root")
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, object]] = []
    for cell in arguments.cells:
        output = arguments.output_root / f"{cell.kind}-{cell.subject}"
        if output.exists():
            raise FileExistsError(f"campaign output already exists: {output}")
        subprocess.run(build_command(arguments, cell, output), check=True)
        completed.append(
            {
                "kind": cell.kind,
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
    arguments: argparse.Namespace, cell: AnchorCell, output: Path
) -> list[str]:
    """Build one guarded GPT or vision anchor command."""
    if cell.kind == "vision":
        module = "benches.spark_vision_point"
        point_arguments = (
            "--subject",
            cell.subject,
            "--image-root",
            str(arguments.image_root),
            "--resolution",
            str(arguments.resolution),
            "--batch-size",
            str(arguments.vision_batch_size),
        )
    else:
        module = "benches.spark_training_point"
        point_arguments = ("--kind", cell.kind, "--subject", cell.subject)
    return guarded_point_command(
        arguments,
        output=output,
        module=module,
        point_arguments=point_arguments,
    )


def _write_summary(
    output_root: Path, completed: list[dict[str, object]], *, status: str
) -> dict[str, object]:
    record = {
        "kind": "spark-training-anchor-campaign",
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
    """Return the runtime-defined named-anchor campaign parser."""
    result = argparse.ArgumentParser()
    result.add_argument("--cell", dest="cells", type=parse_cell, action="append")
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--image-root", type=Path)
    result.add_argument("--resolution", type=int, default=224)
    result.add_argument("--vision-batch-size", type=int, default=64)
    add_spark_runtime_arguments(result)
    return result


def main() -> None:
    """Run a runtime-defined sequence of guarded Spark named anchors."""
    print(run_campaign(parser().parse_args()))


if __name__ == "__main__":
    main()

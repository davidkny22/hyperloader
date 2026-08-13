"""Sequential guarded Spark execution for transformer dial points."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


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
    command = [
        sys.executable,
        "-m",
        "benches.spark_clock_guard",
        "--evidence",
        str(output.with_name(output.name + "-clock.json")),
        "--clock-mhz",
        str(arguments.clock_mhz),
        sys.executable,
        "-m",
        "benches.spark_machine_state_guard",
        "--evidence",
        str(output / "machine-state.json"),
        "--spinner-library",
        str(arguments.spinner_library),
    ]
    for core in arguments.machine_state_cpu:
        command.extend(("--core", str(core)))
    command.extend(
        (
            "--active-us",
            str(arguments.machine_state_active_us),
            "--period-us",
            str(arguments.machine_state_period_us),
            "taskset",
            "-c",
            arguments.cpu_set,
            "env",
            f"PYTHONPATH={arguments.pythonpath}",
            sys.executable,
            "-m",
            "benches.spark_training_point",
            "--kind",
            "dial",
            "--dial-index",
            str(cell.dial_index),
            "--subject",
            cell.subject,
            "--evaluation-id",
            arguments.evaluation_id,
            "--output",
            str(output),
            "--machine",
            arguments.machine,
            "--commit",
            arguments.commit,
            "--lease-token",
            arguments.lease_token,
            "--accelerator-clock",
            arguments.accelerator_clock,
            "--memory-clock",
            arguments.memory_clock,
            "--cpu-governor",
            arguments.cpu_governor,
            "--power-profile",
            arguments.power_profile,
            "--ambient-probe-id",
            arguments.ambient_probe_id,
            "--half-seconds",
            str(arguments.half_seconds),
            "--min-pairs",
            str(arguments.min_pairs),
            "--max-pairs",
            str(arguments.max_pairs),
            "--max-half-width-percent",
            str(arguments.max_half_width_percent),
            "--threshold-percent",
            str(arguments.threshold_percent),
            "--bank-batches",
            str(arguments.bank_batches),
            "--warmup-steps",
            str(arguments.warmup_steps),
            "--bootstrap-draws",
            str(arguments.bootstrap_draws),
            "--tuning-seconds",
            str(arguments.tuning_seconds),
            "--tuning-warmup-steps",
            str(arguments.tuning_warmup_steps),
            "--machine-state-control",
            "native-alu-pulse",
        )
    )
    for candidate in arguments.tuning_candidate:
        command.extend(("--tuning-candidate", candidate))
    for core in arguments.machine_state_cpu:
        command.extend(("--machine-state-cpu", str(core)))
    command.extend(
        (
            "--machine-state-active-us",
            str(arguments.machine_state_active_us),
            "--machine-state-period-us",
            str(arguments.machine_state_period_us),
        )
    )
    return command


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
    result.add_argument("--evaluation-id", required=True)
    result.add_argument("--machine", required=True)
    result.add_argument("--commit", required=True)
    result.add_argument("--lease-token", required=True)
    result.add_argument("--clock-mhz", type=int, required=True)
    result.add_argument("--accelerator-clock", required=True)
    result.add_argument("--memory-clock", required=True)
    result.add_argument("--cpu-governor", required=True)
    result.add_argument("--power-profile", required=True)
    result.add_argument("--ambient-probe-id", required=True)
    result.add_argument("--cpu-set", required=True)
    result.add_argument("--pythonpath", required=True)
    result.add_argument("--spinner-library", type=Path, required=True)
    result.add_argument("--machine-state-cpu", type=int, action="append", required=True)
    result.add_argument("--machine-state-active-us", type=int, required=True)
    result.add_argument("--machine-state-period-us", type=int, required=True)
    result.add_argument("--half-seconds", type=float, default=45.0)
    result.add_argument("--min-pairs", type=int, default=10)
    result.add_argument("--max-pairs", type=int, default=40)
    result.add_argument("--max-half-width-percent", type=float, default=0.15)
    result.add_argument("--threshold-percent", type=float, default=100.0)
    result.add_argument("--bank-batches", type=int, default=64)
    result.add_argument("--warmup-steps", type=int, default=3)
    result.add_argument("--bootstrap-draws", type=int, default=10_000)
    result.add_argument("--tuning-candidate", action="append", required=True)
    result.add_argument("--tuning-seconds", type=float, default=2.0)
    result.add_argument("--tuning-warmup-steps", type=int, default=3)
    return result


def main() -> None:
    """Run a runtime-defined sequence of guarded Spark dial cells."""
    print(run_campaign(parser().parse_args()))


if __name__ == "__main__":
    main()

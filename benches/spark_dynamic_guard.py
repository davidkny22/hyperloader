"""Exclusive Spark dynamic-clock reset around one benchmark command."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def run_guard(*, evidence: Path, command: list[str]) -> None:
    """Release GPU clock constraints before and after one exclusive command."""
    if not command:
        raise ValueError("a benchmark command is required")
    processes = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
    ).stdout.strip()
    if processes:
        raise RuntimeError(f"concurrent GPU processes are present: {processes}")
    reset_command = ["sudo", "-n", "nvidia-smi", "-rgc"]
    initial_reset = _run(reset_command)
    record: dict[str, object] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "clock_mode": "dynamic",
        "initial_reset_stdout": initial_reset.stdout.strip(),
        "command": command,
    }
    command_error: subprocess.CalledProcessError | None = None
    reset_error: subprocess.CalledProcessError | None = None
    try:
        completed = subprocess.run(command, check=True)
        record["command_returncode"] = completed.returncode
    except subprocess.CalledProcessError as error:
        record["command_returncode"] = error.returncode
        command_error = error
    finally:
        try:
            final_reset = _run(reset_command)
            record["final_reset_stdout"] = final_reset.stdout.strip()
        except subprocess.CalledProcessError as error:
            record["final_reset_returncode"] = error.returncode
            record["final_reset_stderr"] = error.stderr
            reset_error = error
        evidence.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if reset_error is not None:
        raise reset_error
    if command_error is not None:
        raise command_error


def main() -> None:
    """Run one command with dynamic GPU clocks and guaranteed final reset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    run_guard(evidence=arguments.evidence, command=arguments.command)


if __name__ == "__main__":
    main()

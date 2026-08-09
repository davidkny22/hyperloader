"""Exclusive Spark GPU clock control around one benchmark command."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def run_guard(*, evidence: Path, clock_mhz: int, command: list[str]) -> None:
    """Run one command under exclusive temporary clock control."""
    if clock_mhz <= 0 or not command:
        raise ValueError("a positive clock and benchmark command are required")
    processes = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
    ).stdout.strip()
    if processes:
        raise RuntimeError(f"concurrent GPU processes are present: {processes}")

    pin = _run(
        [
            "sudo",
            "-n",
            "nvidia-smi",
            "-lgc",
            f"{clock_mhz},{clock_mhz}",
        ]
    )
    record: dict[str, object] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "requested_mhz": clock_mhz,
        "pin_stdout": pin.stdout.strip(),
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
            reset = _run(["sudo", "-n", "nvidia-smi", "-rgc"])
            record["reset_stdout"] = reset.stdout.strip()
        except subprocess.CalledProcessError as error:
            record["reset_returncode"] = error.returncode
            record["reset_stderr"] = error.stderr
            reset_error = error
        evidence.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if reset_error is not None:
        raise reset_error
    if command_error is not None:
        raise command_error


def main() -> None:
    """Apply a requested clock constraint, run a command, and always restore it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--clock-mhz", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    run_guard(
        evidence=arguments.evidence,
        clock_mhz=arguments.clock_mhz,
        command=arguments.command,
    )


if __name__ == "__main__":
    main()

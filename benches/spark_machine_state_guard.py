"""Native machine-state control around one Spark training command."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from benches.dominance_alu_spinner import DutyCycleAluSpinnerGroup


class _Spinner(Protocol):
    def start(self) -> None: ...

    def stop(self) -> dict[str, object]: ...


SpinnerFactory = Callable[..., _Spinner]


def run_guard(
    *,
    evidence: Path,
    spinner_library: Path,
    cores: tuple[int, ...],
    active_microseconds: int,
    period_microseconds: int,
    command: list[str],
    spinner_factory: SpinnerFactory = DutyCycleAluSpinnerGroup,
) -> None:
    """Run one command while native pulses hold its recorded CPU set warm."""
    if not command:
        raise ValueError("a training command is required")
    if not cores or len(set(cores)) != len(cores) or any(core < 0 for core in cores):
        raise ValueError("machine-state CPUs must be a nonempty unique sequence")
    if active_microseconds <= 0 or active_microseconds > period_microseconds:
        raise ValueError("active time must be positive and no longer than the period")
    spinner = spinner_factory(
        spinner_library,
        cores,
        active_microseconds=active_microseconds,
        period_microseconds=period_microseconds,
    )
    record: dict[str, object] = {
        "captured_at": datetime.now(UTC).isoformat(),
        "control": "native-alu-pulse",
        "cores": list(cores),
        "active_microseconds": active_microseconds,
        "period_microseconds": period_microseconds,
        "command": command,
    }
    command_error: subprocess.CalledProcessError | None = None
    stop_error: BaseException | None = None
    spinner.start()
    try:
        completed = subprocess.run(command, check=True)
        record["command_returncode"] = completed.returncode
    except subprocess.CalledProcessError as error:
        record["command_returncode"] = error.returncode
        command_error = error
    finally:
        try:
            record["actuator"] = spinner.stop()
        except BaseException as error:
            record["actuator_stop_error"] = repr(error)
            stop_error = error
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if stop_error is not None:
        raise stop_error
    if command_error is not None:
        raise command_error


def main() -> None:
    """Parse one runtime actuator and run a Spark training command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--spinner-library", type=Path, required=True)
    parser.add_argument("--core", type=int, action="append", required=True)
    parser.add_argument("--active-us", type=int, required=True)
    parser.add_argument("--period-us", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    run_guard(
        evidence=arguments.evidence,
        spinner_library=arguments.spinner_library,
        cores=tuple(arguments.core),
        active_microseconds=arguments.active_us,
        period_microseconds=arguments.period_us,
        command=arguments.command,
    )


if __name__ == "__main__":
    main()

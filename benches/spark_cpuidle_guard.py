"""Scoped Spark CPU-idle state control around one diagnostic command."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

CPU_ROOT = Path("/sys/devices/system/cpu")


def _state_index(path: Path) -> int:
    suffix = path.parent.name.removeprefix("state")
    if not suffix.isdigit():
        raise ValueError(f"invalid CPU-idle state path {path}")
    return int(suffix)


def _target_paths(
    cpus: tuple[int, ...], minimum_state: int, cpu_root: Path = CPU_ROOT
) -> list[Path]:
    if not cpus or len(set(cpus)) != len(cpus):
        raise ValueError("CPU-idle guard CPUs must be a nonempty unique sequence")
    if minimum_state < 1:
        raise ValueError("CPU-idle guard must retain polling state zero")
    targets = []
    for cpu in cpus:
        idle_root = cpu_root / f"cpu{cpu}" / "cpuidle"
        for path in idle_root.glob("state[0-9]*/disable"):
            if _state_index(path) >= minimum_state:
                targets.append(path)
    targets.sort(key=lambda path: (int(path.parents[2].name[3:]), _state_index(path)))
    if not targets:
        raise RuntimeError("CPU-idle guard found no target states")
    return targets


def _read_value(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").strip())


def _write_value(path: Path, value: int) -> str:
    completed = subprocess.run(
        ["sudo", "-n", "tee", str(path)],
        input=f"{value}\n",
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_guard(
    *,
    evidence: Path,
    cpus: tuple[int, ...],
    minimum_state: int,
    command: list[str],
    cpu_root: Path = CPU_ROOT,
) -> None:
    """Disable selected deep states, run one command, and restore every write."""
    if not command:
        raise ValueError("CPU-idle guard requires a diagnostic command")
    targets = _target_paths(cpus, minimum_state, cpu_root)
    initial = {str(path): _read_value(path) for path in targets}
    if any(value != 0 for value in initial.values()):
        raise RuntimeError("CPU-idle guard requires every target state to start enabled")

    writes: list[dict[str, object]] = []
    record: dict[str, object] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "cpus": list(cpus),
        "minimum_state": minimum_state,
        "targets": initial,
        "command": command,
        "writes": writes,
    }
    written: list[Path] = []
    operation_error: BaseException | None = None
    restore_error: Exception | None = None
    try:
        for path in targets:
            stdout = _write_value(path, 1)
            written.append(path)
            writes.append({"path": str(path), "value": 1, "stdout": stdout})
        enabled = {str(path): _read_value(path) for path in targets}
        record["disabled_verification"] = enabled
        if any(value != 1 for value in enabled.values()):
            raise RuntimeError("CPU-idle state disable verification failed")
        completed = subprocess.run(command, check=True)
        record["command_returncode"] = completed.returncode
    except subprocess.CalledProcessError as error:
        record["command_returncode"] = error.returncode
        operation_error = error
    except BaseException as error:
        record["operation_error"] = f"{type(error).__name__}: {error}"
        operation_error = error
    finally:
        restore_failures = []
        for path in reversed(written):
            try:
                stdout = _write_value(path, 0)
                writes.append({"path": str(path), "value": 0, "stdout": stdout})
            except Exception as error:  # noqa: BLE001
                restore_failures.append({"path": str(path), "error": str(error)})
        restored = {str(path): _read_value(path) for path in targets}
        record["restored_verification"] = restored
        record["restore_failures"] = restore_failures
        if restore_failures or any(value != 0 for value in restored.values()):
            restore_error = RuntimeError("CPU-idle state restoration failed")
        evidence.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if restore_error is not None:
        raise restore_error
    if operation_error is not None:
        raise operation_error


def main() -> None:
    """Apply scoped CPU-idle control around one diagnostic command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--cpus", type=int, nargs="+", required=True)
    parser.add_argument("--minimum-state", type=int, default=1)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = list(arguments.command)
    if command[:1] == ["--"]:
        command = command[1:]
    run_guard(
        evidence=arguments.evidence,
        cpus=tuple(arguments.cpus),
        minimum_state=arguments.minimum_state,
        command=command,
    )


if __name__ == "__main__":
    main()

"""Attach py-spy only after a benchmark consumer finishes initialization."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path


def _wait_for_ready(path: Path, child: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        returncode = child.poll()
        if returncode is not None:
            raise RuntimeError(
                f"profile target exited before readiness with code {returncode}"
            )
        time.sleep(0.01)
    raise TimeoutError("profile target did not become ready before the timeout")


def _schedule_profile_start(target_pid: int, delay_seconds: float) -> None:
    helper_pid = os.fork()
    if helper_pid != 0:
        return
    try:
        time.sleep(delay_seconds)
        os.kill(target_pid, signal.SIGUSR1)
    finally:
        os._exit(0)


def main() -> None:
    """Start a target, wait for readiness, then become its sampling parent."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--py-spy", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rate", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--attach-delay-seconds", type=float, default=0.5)
    parser.add_argument("--gil", action="store_true")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--nonblocking", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if not arguments.command:
        raise ValueError("a profile target command is required")
    if arguments.ready_file.exists():
        raise FileExistsError(arguments.ready_file)

    child = subprocess.Popen(arguments.command)
    try:
        _wait_for_ready(arguments.ready_file, child, arguments.timeout_seconds)
        _schedule_profile_start(child.pid, arguments.attach_delay_seconds)
        command = [
            str(arguments.py_spy),
            "record",
            "--format",
            "raw",
            "--threads",
            "--rate",
            str(arguments.rate),
            "--output",
            str(arguments.output),
        ]
        if arguments.gil:
            command.append("--gil")
        if arguments.native:
            command.append("--native")
        if arguments.nonblocking:
            command.append("--nonblocking")
        command.extend(("--pid", str(child.pid)))
        os.execv(command[0], command)
    except BaseException:
        child.terminate()
        child.wait(timeout=10.0)
        raise


if __name__ == "__main__":
    main()

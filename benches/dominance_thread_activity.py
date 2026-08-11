"""Linux consumer-thread activity snapshots for dominance diagnostics."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any


def snapshot_threads() -> dict[int, dict[str, Any]]:
    """Capture task CPU ticks, kernel names, and matching Python thread names."""
    python_names = {
        thread.native_id: thread.name
        for thread in threading.enumerate()
        if thread.native_id is not None
    }
    result: dict[int, dict[str, Any]] = {}
    for task in Path("/proc/self/task").iterdir():
        try:
            stat = _parse_task_stat((task / "stat").read_text(encoding="utf-8"))
        except (FileNotFoundError, ProcessLookupError):
            continue
        task_id = int(task.name)
        result[task_id] = {
            "task_id": task_id,
            **stat,
            "python_name": python_names.get(task_id),
        }
    return result


def diff_thread_cpu(
    before: dict[int, dict[str, Any]], after: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return per-thread CPU milliseconds for tasks alive at phase end."""
    ticks_per_second = float(os.sysconf("SC_CLK_TCK"))
    rows = []
    for task_id, current in after.items():
        previous = before.get(task_id)
        previous_ticks = 0 if previous is None else int(previous["cpu_ticks"])
        delta_ticks = int(current["cpu_ticks"]) - previous_ticks
        rows.append(
            {
                "task_id": task_id,
                "kernel_name": current["kernel_name"],
                "python_name": current["python_name"],
                "cpu_milliseconds": 1000.0 * delta_ticks / ticks_per_second,
                "created_during_phase": previous is None,
            }
        )
    return sorted(rows, key=lambda row: float(row["cpu_milliseconds"]), reverse=True)


def _parse_task_stat(text: str) -> dict[str, int | str]:
    """Parse the comm, state, user ticks, and system ticks from proc task stat."""
    opening = text.find("(")
    closing = text.rfind(")")
    if opening < 0 or closing <= opening:
        raise ValueError("malformed task stat record")
    fields = text[closing + 2 :].split()
    if len(fields) < 13:
        raise ValueError("incomplete task stat record")
    return {
        "kernel_name": text[opening + 1 : closing],
        "state": fields[0],
        "user_ticks": int(fields[11]),
        "system_ticks": int(fields[12]),
        "cpu_ticks": int(fields[11]) + int(fields[12]),
    }

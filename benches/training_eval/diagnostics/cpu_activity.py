"""Per-core Linux CPU activity for bounded training diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def snapshot_cpu_activity(path: Path = Path("/proc/stat")) -> dict[str, tuple[int, ...]]:
    """Capture cumulative per-core scheduler ticks."""
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("cpu") or not fields[0][3:].isdigit():
            continue
        result[fields[0]] = tuple(int(value) for value in fields[1:])
    return result


def diff_cpu_activity(
    before: dict[str, tuple[int, ...]], after: dict[str, tuple[int, ...]]
) -> list[dict[str, Any]]:
    """Return active, idle, and utilization deltas for every surviving core."""
    rows = []
    for cpu, current in after.items():
        previous = before.get(cpu)
        if previous is None or len(previous) != len(current):
            continue
        deltas = tuple(stop - start for start, stop in zip(previous, current, strict=True))
        total = sum(deltas)
        idle = sum(deltas[index] for index in (3, 4) if index < len(deltas))
        active = total - idle
        rows.append(
            {
                "cpu": cpu,
                "active_ticks": active,
                "idle_ticks": idle,
                "total_ticks": total,
                "utilization_percent": 0.0 if total <= 0 else 100.0 * active / total,
            }
        )
    return sorted(rows, key=lambda row: int(str(row["cpu"])[3:]))

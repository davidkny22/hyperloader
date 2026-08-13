"""Operating-system and accelerator-state capture."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


def cpu_governors() -> dict[str, str] | str:
    """Return every readable CPU frequency governor."""
    paths = sorted(
        Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")
    )
    if not paths:
        return "unavailable"
    return {path.parts[-3]: _read_text(path) for path in paths}


def cpu_clocks() -> dict[str, str] | str:
    """Return every readable current CPU frequency."""
    paths = sorted(
        Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq")
    )
    if not paths:
        return "unavailable"
    return {path.parts[-3]: _read_text(path) for path in paths}


def idle_states() -> dict[str, dict[str, dict[str, str]]] | str:
    """Return idle-state configuration and cumulative residency counters."""
    roots = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpuidle/state*"))
    if not roots:
        return "unavailable"
    values: dict[str, dict[str, dict[str, str]]] = {}
    for root in roots:
        cpu = root.parts[-3]
        values.setdefault(cpu, {})[root.name] = {
            field: _read_text(root / field)
            for field in ("name", "disable", "time", "usage")
        }
    return values


def interrupt_homes() -> dict[str, Any] | str:
    """Return accelerator interrupt counters per CPU."""
    path = Path("/proc/interrupts")
    if not path.exists():
        return "unavailable"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return "unavailable"
    cpu_count = sum(token.startswith("CPU") for token in lines[0].split())
    records = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) <= cpu_count + 1:
            continue
        label = " ".join(fields[cpu_count + 1 :])
        if not any(token in label.lower() for token in ("nvidia", "gpu", "nvhost")):
            continue
        records.append(
            {
                "counts": [int(value) for value in fields[1 : cpu_count + 1]],
                "irq": fields[0].rstrip(":"),
                "label": label,
            }
        )
    return {"cpu_count": cpu_count, "records": records}


def thermal_zones() -> dict[str, str] | str:
    """Return readable platform temperatures."""
    paths = sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp"))
    if not paths:
        return "unavailable"
    return {path.parent.name: _read_text(path) for path in paths}


def background_load() -> dict[str, Any]:
    """Return host load and accelerator client state."""
    return {
        "accelerator_clients": nvidia_compute_apps(),
        "load_average": list(os.getloadavg())
        if hasattr(os, "getloadavg")
        else "unavailable",
        "process_count": _process_count(),
    }


def memory_pressure() -> dict[str, str] | str:
    """Return host memory and file-cache totals."""
    path = Path("/proc/meminfo")
    if not path.exists():
        return "unavailable"
    wanted = {
        "Active(file)",
        "Buffers",
        "Cached",
        "MemAvailable",
        "MemFree",
        "MemTotal",
        "SwapFree",
        "SwapTotal",
    }
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition(":")
        if name in wanted:
            values[name] = value.strip()
    return values


def storage_state() -> dict[str, Any]:
    """Return filesystem identity and free-space state without embedding a path."""
    current = Path.cwd()
    try:
        usage = os.statvfs(current)
    except (AttributeError, OSError):
        return {"working_directory_device": "unavailable"}
    return {
        "filesystem_block_size": usage.f_frsize,
        "filesystem_free_bytes": usage.f_bavail * usage.f_frsize,
        "working_directory_device": current.stat().st_dev,
    }


def nvidia_query(fields: tuple[str, ...]) -> list[dict[str, str]] | str:
    """Return one parsed nvidia-smi query or a labeled unavailable value."""
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        return f"unavailable: {type(error).__name__}"
    rows = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        rows.append(dict(zip(fields, values, strict=False)))
    return rows


def nvidia_compute_apps() -> list[dict[str, str]] | str:
    """Return active accelerator compute clients."""
    fields = ("pid", "process_name", "used_memory")
    command = [
        "nvidia-smi",
        f"--query-compute-apps={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        return f"unavailable: {type(error).__name__}"
    rows = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        rows.append(dict(zip(fields, values, strict=False)))
    return rows


def affinity(pid: int) -> list[int] | str:
    """Return one process CPU mask."""
    if not hasattr(os, "sched_getaffinity"):
        return "unavailable"
    try:
        return sorted(os.sched_getaffinity(pid))
    except OSError:
        return "unavailable"


def thread_count(status: str | Path) -> int | str:
    """Return one Linux process thread count."""
    path = Path(status)
    if not path.exists():
        return "unavailable"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Threads:"):
            return int(line.split(":", 1)[1].strip())
    return "unavailable"


def _process_count() -> int | str:
    root = Path("/proc")
    if not root.exists():
        return "unavailable"
    return sum(path.name.isdigit() for path in root.iterdir())


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as error:
        return f"unavailable: {type(error).__name__}"

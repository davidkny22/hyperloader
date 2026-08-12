"""Cross-platform worker process resource snapshots."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    class _FileTime(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

        def ticks(self) -> int:
            return (int(self.high) << 32) | int(self.low)

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("page_fault_count", wintypes.DWORD),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_nonpaged_pool_usage", ctypes.c_size_t),
            ("quota_nonpaged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
        ]

    _KERNEL = ctypes.windll.kernel32
    _PSAPI = ctypes.windll.psapi
    _KERNEL.OpenProcess.restype = wintypes.HANDLE


def snapshot_workers(workers: tuple[Any, ...]) -> list[dict[str, int | bool | None]]:
    """Read CPU time and resident bytes without signaling worker processes."""
    return [
        {"worker": index, **_snapshot_process(process)}
        for index, process in enumerate(workers)
    ]


def _snapshot_process(process: Any) -> dict[str, int | bool | None]:
    pid = process if isinstance(process, int) else getattr(process, "pid", None)
    if not isinstance(pid, int):
        return {"pid": 0, "alive": False, "cpu_ns": None, "rss_bytes": None}
    if os.name == "nt":
        handle = (
            None if isinstance(process, int) else getattr(process, "sentinel", None)
        )
        return _snapshot_windows(pid, handle)
    return _snapshot_procfs(pid)


def _snapshot_procfs(pid: int) -> dict[str, int | bool | None]:
    root = Path("/proc") / str(pid)
    try:
        stat = (root / "stat").read_text(encoding="utf-8")
        fields = stat[stat.rfind(")") + 2 :].split()
        ticks = int(fields[11]) + int(fields[12])
        ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
        cpu_ns = ticks * 1_000_000_000 // ticks_per_second
        rss_bytes = _procfs_rss(root / "status")
    except (IndexError, OSError, ValueError):
        return {"pid": pid, "alive": False, "cpu_ns": None, "rss_bytes": None}
    return {"pid": pid, "alive": True, "cpu_ns": cpu_ns, "rss_bytes": rss_bytes}


def _procfs_rss(path: Path) -> int | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def _snapshot_windows(
    pid: int, borrowed_handle: int | None = None
) -> dict[str, int | bool | None]:
    handle = borrowed_handle or _KERNEL.OpenProcess(0x1000, False, pid)
    if not handle:
        return {"pid": pid, "alive": False, "cpu_ns": None, "rss_bytes": None}
    try:
        created, exited, kernel_time, user_time = (_FileTime() for _ in range(4))
        cpu_ok = _KERNEL.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        )
        memory = _ProcessMemoryCounters()
        memory.cb = ctypes.sizeof(memory)
        rss_ok = _PSAPI.GetProcessMemoryInfo(
            handle, ctypes.byref(memory), ctypes.sizeof(memory)
        )
        return {
            "pid": pid,
            "alive": True,
            "cpu_ns": (
                (kernel_time.ticks() + user_time.ticks()) * 100 if cpu_ok else None
            ),
            "rss_bytes": int(memory.working_set_size) if rss_ok else None,
        }
    finally:
        if borrowed_handle is None:
            _KERNEL.CloseHandle(handle)

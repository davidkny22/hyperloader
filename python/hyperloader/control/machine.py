"""Stable machine identity for calibration and profile cache keys."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CpuCluster:
    """One group of logical CPUs with a shared capacity class."""

    name: str
    logical_cpus: tuple[int, ...]
    max_frequency_hz: int | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.logical_cpus:
            raise ValueError("CPU clusters require a name and at least one logical CPU")
        if len(set(self.logical_cpus)) != len(self.logical_cpus):
            raise ValueError("CPU cluster logical CPUs must be unique")
        if self.max_frequency_hz is not None and self.max_frequency_hz <= 0:
            raise ValueError("CPU cluster maximum frequency must be positive")


@dataclass(frozen=True, slots=True)
class MachineIdentity:
    """Hardware identity that invalidates machine-specific cached decisions."""

    cpu_model: str
    clusters: tuple[CpuCluster, ...]
    memory_bytes: int

    def __post_init__(self) -> None:
        if not self.cpu_model.strip():
            raise ValueError("machine identity requires a CPU model")
        if not self.clusters:
            raise ValueError("machine identity requires CPU topology")
        if self.memory_bytes <= 0:
            raise ValueError("machine identity memory must be positive")
        cpus = [cpu for cluster in self.clusters for cpu in cluster.logical_cpus]
        if len(cpus) != len(set(cpus)):
            raise ValueError("logical CPUs cannot appear in multiple clusters")

    @property
    def cache_key(self) -> str:
        """Return the canonical SHA-256 hardware digest."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return the canonical hardware payload."""
        return {
            "clusters": [
                {
                    "logical_cpus": list(cluster.logical_cpus),
                    "max_frequency_hz": cluster.max_frequency_hz,
                    "name": cluster.name,
                }
                for cluster in self.clusters
            ],
            "cpu_model": self.cpu_model,
            "memory_bytes": self.memory_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> MachineIdentity:
        """Validate and construct a machine identity payload."""
        raw_clusters = payload.get("clusters")
        if not isinstance(raw_clusters, list):
            raise ValueError("machine identity clusters must be a list")
        clusters = []
        for raw in raw_clusters:
            if not isinstance(raw, dict):
                raise ValueError("machine identity cluster must be an object")
            cpus = raw.get("logical_cpus")
            if not isinstance(cpus, list) or not all(isinstance(cpu, int) for cpu in cpus):
                raise ValueError("machine identity logical CPUs must be integers")
            frequency = raw.get("max_frequency_hz")
            if frequency is not None and not isinstance(frequency, int):
                raise ValueError("machine identity frequency must be an integer or null")
            clusters.append(CpuCluster(str(raw.get("name", "")), tuple(cpus), frequency))
        return cls(
            cpu_model=str(payload.get("cpu_model", "")),
            clusters=tuple(clusters),
            memory_bytes=int(payload.get("memory_bytes", 0)),
        )


def detect_machine_identity() -> MachineIdentity:
    """Measure the local CPU model, capacity topology, and physical memory."""
    return MachineIdentity(_cpu_model(), _cpu_clusters(), _physical_memory_bytes())


def _cpu_model() -> str:
    model = platform.processor().strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith(("model name", "hardware")) and ":" in line:
                model = line.split(":", 1)[1].strip()
                if model:
                    break
    return model or platform.machine() or "unknown-cpu"


def _cpu_clusters() -> tuple[CpuCluster, ...]:
    count = os.cpu_count() or 1
    frequencies: dict[int | None, list[int]] = {}
    for cpu in range(count):
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq")
        frequency = int(path.read_text(encoding="ascii").strip()) * 1000 if path.is_file() else None
        frequencies.setdefault(frequency, []).append(cpu)
    ordered = sorted(frequencies.items(), key=lambda item: item[0] or 0)
    if len(ordered) == 1:
        frequency, cpus = ordered[0]
        return (CpuCluster("all", tuple(cpus), frequency),)
    return tuple(
        CpuCluster(
            "efficiency" if index == 0 else "performance" if index == len(ordered) - 1 else f"capacity-{index}",
            tuple(cpus),
            frequency,
        )
        for index, (frequency, cpus) in enumerate(ordered)
    )


def _physical_memory_bytes() -> int:
    if os.name != "nt":
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("physical memory measurement failed")
    return int(status.total_physical)

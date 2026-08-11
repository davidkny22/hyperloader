"""Spark environment capture and under-load GPU clock sampling."""

from __future__ import annotations

import os
import platform
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClockSample:
    """One observed GPU clock and utilization point."""

    elapsed_seconds: float
    clock_mhz: int
    memory_clock_mhz: int | None
    utilization_percent: int
    power_watts: float | None
    spark_hwmon: dict[str, float]


class ClockSampler:
    """Capture low-rate clock evidence without sharing the workload core."""

    def __init__(self, interval_seconds: float = 1.0, *, affinity_cpu: int = 5) -> None:
        self._interval = interval_seconds
        self._affinity_cpu = affinity_cpu
        self._stop = threading.Event()
        self._samples: list[ClockSample] = []
        self._started = 0.0
        self._thread = threading.Thread(
            target=self._run, name="clock-sampler", daemon=True
        )
        self._rail_sources = _spark_hwmon_sources()

    @property
    def rail_sources(self) -> dict[str, str]:
        """Return readable Spark rail labels and their sysfs sources."""
        return {label: str(path) for label, path in self._rail_sources.items()}

    def elapsed_seconds(self) -> float:
        """Return elapsed time on the sampler's measurement clock."""
        if self._started == 0.0:
            raise RuntimeError("clock sampler has not started")
        return time.perf_counter() - self._started

    def start(self) -> None:
        """Start sampling from a dedicated efficiency core."""
        self._started = time.perf_counter()
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        """Stop sampling and return JSON-compatible raw observations."""
        self._stop.set()
        self._thread.join()
        return [asdict(sample) for sample in self._samples]

    def _run(self) -> None:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {self._affinity_cpu})
        while not self._stop.is_set():
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=clocks.current.sm,clocks.current.memory,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            clock, memory_clock, utilization, power = completed.stdout.strip().split(
                ", "
            )
            self._samples.append(
                ClockSample(
                    elapsed_seconds=time.perf_counter() - self._started,
                    clock_mhz=int(clock),
                    memory_clock_mhz=_optional_int(memory_clock),
                    utilization_percent=int(utilization),
                    power_watts=_optional_float(power),
                    spark_hwmon={
                        label: float(path.read_text(encoding="utf-8").strip())
                        for label, path in self._rail_sources.items()
                    },
                )
            )
            self._stop.wait(self._interval)


def hardware_monitor_facts() -> dict[str, Any]:
    """Describe readable machine-state instruments without changing the system."""
    sources = _spark_hwmon_sources()
    bpmp_path = Path("/sys/kernel/debug/bpmp")
    try:
        bpmp_path.stat()
        bpmp_present: bool | None = True
        bpmp_probe_error = None
    except FileNotFoundError:
        bpmp_present = False
        bpmp_probe_error = None
    except PermissionError as error:
        bpmp_present = None
        bpmp_probe_error = f"{type(error).__name__}: {error}"
    return {
        "spark_hwmon_available": bool(sources),
        "spark_hwmon_sources": {label: str(path) for label, path in sources.items()},
        "bpmp_path": str(bpmp_path),
        "bpmp_present": bpmp_present,
        "bpmp_probe_error": bpmp_probe_error,
    }


def _spark_hwmon_sources() -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for directory in Path("/sys/class/hwmon").glob("hwmon*"):
        name_path = directory / "name"
        try:
            if name_path.read_text(encoding="utf-8").strip() != "spark_hwmon":
                continue
        except OSError:
            continue
        for pattern in ("power*_input", "in*_input", "curr*_input"):
            for path in directory.glob(pattern):
                if not os.access(path, os.R_OK):
                    continue
                stem = path.name.removesuffix("_input")
                label_path = directory / f"{stem}_label"
                try:
                    label = label_path.read_text(encoding="utf-8").strip()
                except OSError:
                    label = stem
                sources[f"{label}:{path.name}"] = path
    return sources


def _optional_int(value: str) -> int | None:
    return None if value == "[N/A]" else int(value)


def _optional_float(value: str) -> float | None:
    return None if value == "[N/A]" else float(value)


def cpu_governor() -> str:
    """Return the common observed CPU governor or reject mixed controls."""
    governors = {
        path.read_text(encoding="utf-8").strip()
        for path in Path("/sys/devices/system/cpu").glob(
            "cpu*/cpufreq/scaling_governor"
        )
    }
    if len(governors) != 1:
        raise RuntimeError(f"CPU governors are not uniform: {sorted(governors)}")
    return governors.pop()


def total_llc_bytes() -> int:
    """Sum each distinct last-level cache instance reported by Linux sysfs."""
    instances: dict[str, int] = {}
    for index in Path("/sys/devices/system/cpu").glob("cpu*/cache/index*"):
        if (index / "level").read_text(encoding="utf-8").strip() != "3":
            continue
        shared = (index / "shared_cpu_list").read_text(encoding="utf-8").strip()
        size = (index / "size").read_text(encoding="utf-8").strip()
        multiplier = 1024 if size.endswith("K") else 1024 * 1024
        instances[shared] = int(size[:-1]) * multiplier
    if not instances:
        raise RuntimeError("Linux sysfs reported no last-level cache instances")
    return sum(instances.values())


def platform_facts() -> dict[str, str]:
    """Capture immutable host facts visible inside the benchmark container."""
    return {
        "operating_system": platform.system(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
    }

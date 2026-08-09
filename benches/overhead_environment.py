"""Spark environment capture and under-load GPU clock sampling."""

from __future__ import annotations

import os
import platform
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClockSample:
    """One observed GPU clock and utilization point."""

    elapsed_seconds: float
    clock_mhz: int
    utilization_percent: int


class ClockSampler:
    """Capture low-rate clock evidence without sharing the workload core."""

    def __init__(self, interval_seconds: float = 1.0) -> None:
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._samples: list[ClockSample] = []
        self._started = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """Start sampling from a dedicated efficiency core."""
        self._started = time.perf_counter()
        self._thread.start()

    def stop(self) -> list[dict[str, float | int]]:
        """Stop sampling and return JSON-compatible raw observations."""
        self._stop.set()
        self._thread.join()
        return [asdict(sample) for sample in self._samples]

    def _run(self) -> None:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {5})
        while not self._stop.is_set():
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=clocks.current.sm,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            clock, utilization = completed.stdout.strip().split(", ")
            self._samples.append(
                ClockSample(
                    elapsed_seconds=time.perf_counter() - self._started,
                    clock_mhz=int(clock),
                    utilization_percent=int(utilization),
                )
            )
            self._stop.wait(self._interval)


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

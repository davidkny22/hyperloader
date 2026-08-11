"""Linux CPU-idle and GPU-interrupt counter attribution helpers."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path


class HalfBoundaryCpuIdleSampler:
    """Capture CPU-idle counters around two uninterrupted feeder halves."""

    def __init__(self, *, affinity_cpu: int = 14) -> None:
        self._affinity_cpu = affinity_cpu
        self._before: dict[str, object] | None = None
        self._midpoint: dict[str, object] | None = None
        self._error: BaseException | None = None
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, midpoint_seconds: float) -> None:
        """Capture the opening boundary and schedule the midpoint capture."""
        if self._thread is not None:
            raise RuntimeError("CPU-idle boundary sampler is already running")
        self._before = snapshot_cpuidle()
        self._thread = threading.Thread(
            target=self._capture_midpoint,
            args=(midpoint_seconds,),
            name="dominance-cpuidle-boundary",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, object]:
        """Capture the closing boundary and return both half deltas."""
        thread = self._thread
        if thread is None or self._before is None:
            raise RuntimeError("CPU-idle boundary sampler is not running")
        thread.join()
        if self._error is not None:
            raise RuntimeError("CPU-idle midpoint capture failed") from self._error
        if self._midpoint is None:
            raise RuntimeError("CPU-idle midpoint capture was cancelled")
        after = snapshot_cpuidle()
        first_seconds = _snapshot_interval_seconds(self._before, self._midpoint)
        second_seconds = _snapshot_interval_seconds(self._midpoint, after)
        return {
            "first": diff_cpuidle(self._before, self._midpoint, first_seconds),
            "second": diff_cpuidle(self._midpoint, after, second_seconds),
        }

    def close(self) -> None:
        """Cancel an unfinished midpoint capture during exceptional cleanup."""
        self._cancel.set()
        if self._thread is not None:
            self._thread.join()

    def _capture_midpoint(self, midpoint_seconds: float) -> None:
        try:
            if hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, {self._affinity_cpu})
            remaining = max(0.0, midpoint_seconds - time.perf_counter())
            if self._cancel.wait(remaining):
                return
            self._midpoint = snapshot_cpuidle()
        except (OSError, RuntimeError, ValueError) as error:
            self._error = error


def snapshot_cpuidle(
    cpu_root: Path = Path("/sys/devices/system/cpu"),
) -> dict[str, object]:
    """Capture every per-CPU idle-state property and cumulative counter."""
    cpus: dict[str, object] = {}
    for cpu_path in sorted(
        cpu_root.glob("cpu[0-9]*"), key=lambda path: _numeric_suffix(path.name, "cpu")
    ):
        idle_root = cpu_path / "cpuidle"
        if not idle_root.is_dir():
            continue
        states: dict[str, object] = {}
        for state_path in sorted(
            idle_root.glob("state[0-9]*"),
            key=lambda path: _numeric_suffix(path.name, "state"),
        ):
            state = str(_numeric_suffix(state_path.name, "state"))
            states[state] = {
                "name": _read_text(state_path / "name"),
                "description": _read_text(state_path / "desc"),
                "exit_latency_us": _read_int(state_path / "latency"),
                "target_residency_us": _read_int(state_path / "residency"),
                "power_mw": _read_int(state_path / "power"),
                "disabled": bool(_read_int(state_path / "disable")),
                "time_us": _read_int(state_path / "time"),
                "usage": _read_int(state_path / "usage"),
            }
        cpus[str(_numeric_suffix(cpu_path.name, "cpu"))] = states
    if not cpus:
        raise RuntimeError(f"no CPU-idle counters found under {cpu_root}")
    return {"captured_monotonic_ns": time.monotonic_ns(), "cpus": cpus}


def diff_cpuidle(
    before: dict[str, object],
    after: dict[str, object],
    duration_seconds: float,
) -> dict[str, object]:
    """Subtract two CPU-idle snapshots without hiding any CPU or state."""
    if duration_seconds <= 0:
        raise ValueError("CPU-idle attribution duration must be positive")
    before_cpus = dict(before["cpus"])
    after_cpus = dict(after["cpus"])
    if before_cpus.keys() != after_cpus.keys():
        raise RuntimeError("CPU-idle CPU inventory changed during measurement")
    rows = []
    for cpu in sorted(before_cpus, key=int):
        before_states = dict(before_cpus[cpu])
        after_states = dict(after_cpus[cpu])
        if before_states.keys() != after_states.keys():
            raise RuntimeError(f"CPU-idle state inventory changed on CPU{cpu}")
        for state in sorted(before_states, key=int):
            earlier = dict(before_states[state])
            later = dict(after_states[state])
            time_delta = int(later["time_us"]) - int(earlier["time_us"])
            usage_delta = int(later["usage"]) - int(earlier["usage"])
            if time_delta < 0 or usage_delta < 0:
                raise RuntimeError(
                    f"CPU-idle counters moved backward on CPU{cpu} state{state}"
                )
            rows.append(
                {
                    "cpu": int(cpu),
                    "state": int(state),
                    "name": later["name"],
                    "description": later["description"],
                    "exit_latency_us": int(later["exit_latency_us"]),
                    "target_residency_us": int(later["target_residency_us"]),
                    "time_delta_us": time_delta,
                    "usage_delta": usage_delta,
                    "window_residency_percent": 100.0
                    * time_delta
                    / (duration_seconds * 1_000_000.0),
                    "mean_requested_residency_us": (
                        time_delta / usage_delta if usage_delta else None
                    ),
                }
            )
    return {"duration_seconds": duration_seconds, "rows": rows}


def snapshot_gpu_interrupts(
    interrupts_path: Path = Path("/proc/interrupts"),
) -> dict[str, object]:
    """Capture every NVIDIA-labeled hardware interrupt count by CPU."""
    return parse_gpu_interrupts(interrupts_path.read_text(encoding="utf-8"))


def parse_gpu_interrupts(text: str) -> dict[str, object]:
    """Parse the CPU header and NVIDIA-labeled rows from `/proc/interrupts`."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("interrupt inventory is empty")
    cpus = lines[0].split()
    if not cpus or any(not cpu.startswith("CPU") for cpu in cpus):
        raise RuntimeError("interrupt inventory has no CPU header")
    rows: dict[str, object] = {}
    for line in lines[1:]:
        if "nvidia" not in line.casefold() or ":" not in line:
            continue
        irq, remainder = line.split(":", 1)
        fields = remainder.split()
        if len(fields) < len(cpus):
            raise RuntimeError(f"interrupt row {irq.strip()} has too few CPU counts")
        counts = [int(value) for value in fields[: len(cpus)]]
        rows[irq.strip()] = {
            "counts": dict(zip(cpus, counts, strict=True)),
            "label": " ".join(fields[len(cpus) :]),
        }
    if not rows:
        raise RuntimeError("no NVIDIA interrupt rows found")
    return {
        "captured_monotonic_ns": time.monotonic_ns(),
        "cpus": cpus,
        "rows": rows,
    }


def diff_gpu_interrupts(
    before: dict[str, object], after: dict[str, object]
) -> dict[str, object]:
    """Subtract NVIDIA interrupt snapshots by IRQ and CPU."""
    cpus = list(before["cpus"])
    if cpus != list(after["cpus"]):
        raise RuntimeError("interrupt CPU inventory changed during measurement")
    before_rows = dict(before["rows"])
    after_rows = dict(after["rows"])
    if before_rows.keys() != after_rows.keys():
        raise RuntimeError("NVIDIA interrupt inventory changed during measurement")
    rows = []
    totals = {cpu: 0 for cpu in cpus}
    for irq in sorted(before_rows, key=_irq_sort_key):
        earlier = dict(before_rows[irq])
        later = dict(after_rows[irq])
        earlier_counts = dict(earlier["counts"])
        later_counts = dict(later["counts"])
        deltas = {
            cpu: int(later_counts[cpu]) - int(earlier_counts[cpu]) for cpu in cpus
        }
        if any(value < 0 for value in deltas.values()):
            raise RuntimeError(f"interrupt counts moved backward on IRQ {irq}")
        for cpu, value in deltas.items():
            totals[cpu] += value
        rows.append({"irq": irq, "label": later["label"], "deltas": deltas})
    return {"cpus": cpus, "rows": rows, "aggregate_deltas": totals}


def capture_kernel_counters() -> dict[str, object]:
    """Capture CPU-idle and GPU-interrupt counters at one measurement boundary."""
    started = time.monotonic_ns()
    cpuidle = snapshot_cpuidle()
    interrupts = snapshot_gpu_interrupts()
    finished = time.monotonic_ns()
    return {
        "capture_started_monotonic_ns": started,
        "capture_finished_monotonic_ns": finished,
        "capture_duration_ms": (finished - started) / 1_000_000.0,
        "cpuidle": cpuidle,
        "gpu_interrupts": interrupts,
    }


def _numeric_suffix(value: str, prefix: str) -> int:
    suffix = value.removeprefix(prefix)
    if not suffix.isdigit():
        raise ValueError(f"{value} has no numeric {prefix} suffix")
    return int(suffix)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _read_int(path: Path) -> int:
    return int(_read_text(path))


def _snapshot_interval_seconds(
    before: dict[str, object], after: dict[str, object]
) -> float:
    elapsed_ns = int(after["captured_monotonic_ns"]) - int(
        before["captured_monotonic_ns"]
    )
    if elapsed_ns <= 0:
        raise RuntimeError("CPU-idle snapshot clock did not advance")
    return elapsed_ns / 1_000_000_000.0


def _irq_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)

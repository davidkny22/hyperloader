"""Measure enabled telemetry against the disabled public loader path."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

from hyperloader import DataLoader, HyperConfig, _hyperloader
from hyperloader.config import TelemetryConfig

from telemetry_overhead_report import BATCH_SIZE, MINIMUM_PAIRS, TARGET_SAMPLE_RATE


def _pin_process(cpu: int | None) -> None:
    if cpu is None:
        return
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
        return
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        if not kernel32.SetProcessAffinityMask(
            kernel32.GetCurrentProcess(), ctypes.c_size_t(1 << cpu)
        ):
            raise OSError("failed to set the benchmark process affinity")
        return
    raise OSError("process affinity is unavailable on this platform")


def _run_half(
    dataset: torch.Tensor, enabled: bool, pace_ns: int = 0
) -> dict[str, Any]:
    config = HyperConfig(telemetry=TelemetryConfig(enabled=enabled))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2, config=config)
    checksum = 0
    batches = 0
    started_cpu = time.process_time_ns()
    started_wall = time.perf_counter_ns()
    deadline_ns = started_wall
    try:
        for batch in loader:
            checksum += int(batch[0].item()) + int(batch[-1].item())
            batches += 1
            if pace_ns:
                deadline_ns += pace_ns
                remaining_ns = deadline_ns - time.perf_counter_ns()
                if remaining_ns > 200_000:
                    time.sleep((remaining_ns - 100_000) / 1e9)
                while time.perf_counter_ns() < deadline_ns:
                    pass
        wall_ns = time.perf_counter_ns() - started_wall
        cpu_ns = time.process_time_ns() - started_cpu
        snapshot = loader.stats()
    finally:
        loader.close()
    if enabled:
        summary = snapshot["last_epoch"]
        if not snapshot["enabled"] or summary["delivered_batches"] != batches:
            raise AssertionError("enabled public path did not publish its delivery summary")
        if summary["delivered_samples"] != dataset.shape[0]:
            raise AssertionError("enabled public path reported the wrong sample count")
    elif snapshot["enabled"] or snapshot["current"] is not None:
        raise AssertionError("disabled public path unexpectedly allocated telemetry")
    return {
        "batches": batches,
        "checksum": checksum,
        "cpu_ns": cpu_ns,
        "wall_ns": wall_ns,
    }


def _pair(
    dataset: torch.Tensor,
    left_enabled: bool,
    right_enabled: bool,
    order: str,
    pace_ns: int = 0,
) -> dict[str, Any]:
    left = _run_half(dataset, left_enabled, pace_ns)
    right = _run_half(dataset, right_enabled, pace_ns)
    if left["batches"] != right["batches"]:
        raise AssertionError("paired executions delivered different batch counts")
    return {
        "left_checksum": left["checksum"],
        "left_cpu_ns": left["cpu_ns"],
        "left_wall_ns": left["wall_ns"],
        "order": order,
        "right_checksum": right["checksum"],
        "right_cpu_ns": right["cpu_ns"],
        "right_wall_ns": right["wall_ns"],
    }


def run_measurement(
    pair_count: int,
    cpu_batches: int,
    wall_batches: int,
    expected_root: Path,
    cpu: int | None,
) -> dict[str, Any]:
    """Run paired telemetry and null cells through an installed artifact."""
    if pair_count < MINIMUM_PAIRS:
        raise ValueError("pair count is below the measurement floor")
    if cpu_batches <= 0 or wall_batches <= 0:
        raise ValueError("CPU and wall batches must be positive")
    extension_path = Path(_hyperloader.__file__).resolve()
    if not extension_path.is_relative_to(expected_root.resolve()):
        raise RuntimeError("benchmark did not import the expected installed artifact")
    _pin_process(cpu)
    torch.set_num_threads(1)
    cpu_dataset = torch.arange(cpu_batches * BATCH_SIZE, dtype=torch.int64)
    wall_dataset = torch.arange(wall_batches * BATCH_SIZE, dtype=torch.int64)
    pace_ns = round(1e9 * BATCH_SIZE / TARGET_SAMPLE_RATE)
    _run_half(cpu_dataset, False)
    _run_half(cpu_dataset, True)
    _run_half(wall_dataset, False, pace_ns)
    _run_half(wall_dataset, True, pace_ns)
    cpu_pairs = []
    wall_pairs = []
    noise_pairs = []
    for index in range(pair_count):
        enabled_first = index % 2 == 0
        order = "enabled-first" if enabled_first else "disabled-first"
        cpu_pairs.append(
            _pair(
                cpu_dataset,
                enabled_first,
                not enabled_first,
                order,
            )
        )
        wall_pairs.append(
            _pair(
                wall_dataset,
                enabled_first,
                not enabled_first,
                order,
                pace_ns,
            )
        )
        noise_pairs.append(_pair(wall_dataset, False, False, "null", pace_ns))
    return {
        "metadata": {
            "batch_size": BATCH_SIZE,
            "cpu_batches_per_half": cpu_batches,
            "extension_path": str(extension_path),
            "pair_count": pair_count,
            "pace_ns": pace_ns,
            "platform": platform.platform(),
            "process_clock_resolution_ns": round(time.get_clock_info("process_time").resolution * 1e9),
            "public_path_verified": True,
            "python": sys.version,
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "telemetry_summary_verified": True,
            "torch": torch.__version__,
            "wall_batches_per_half": wall_batches,
        },
        "cpu_pairs": cpu_pairs,
        "noise_pairs": noise_pairs,
        "wall_pairs": wall_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=MINIMUM_PAIRS)
    parser.add_argument("--cpu-batches", type=int, default=65_536)
    parser.add_argument("--wall-batches", type=int, default=512)
    parser.add_argument("--expected-install-root", type=Path, required=True)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run_measurement(
        arguments.pairs,
        arguments.cpu_batches,
        arguments.wall_batches,
        arguments.expected_install_root,
        arguments.cpu,
    )
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

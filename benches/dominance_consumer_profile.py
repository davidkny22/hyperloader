"""Paired fixed-text consumer target for external GIL and native profiling."""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dominance_alu_spinner import AluSpinner
from dominance_feeders import build_feeder
from dominance_gpu_segments import summarize_clocks
from dominance_protocol import SelectedConfig
from dominance_thread_activity import diff_thread_cpu, snapshot_threads
from dominance_workloads import make_workload
from overhead_environment import ClockSampler, cpu_governor, hardware_monitor_facts
from overhead_workload import GpuWorkload


def profile_next_batch(feeder: Any) -> Any:
    """Expose feeder activity as a stable sampling-profiler frame."""
    return feeder.next_batch()


def profile_copy(workload: GpuWorkload, batch: Any) -> Any:
    """Expose the synchronous consumer transfer as a profiler frame."""
    return batch.to("cuda", non_blocking=False)


def profile_launch(workload: GpuWorkload, device_batch: Any) -> None:
    """Expose kernel launch activity as a profiler frame."""
    workload._run_kernels(device_batch)


def profile_sync(workload: GpuWorkload) -> None:
    """Expose consumer synchronization as a profiler frame."""
    workload._torch.cuda.synchronize()


def _profile_half(
    feeder: Any, workload: GpuWorkload, seconds: float, sampler: ClockSampler
) -> dict[str, object]:
    before = snapshot_threads()
    iterations = 0
    started = sampler.elapsed_seconds()
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        batch = profile_next_batch(feeder)
        device_batch = profile_copy(workload, batch)
        profile_launch(workload, device_batch)
        profile_sync(workload)
        iterations += 1
    finished = sampler.elapsed_seconds()
    after = snapshot_threads()
    return {
        "start_seconds": started,
        "end_seconds": finished,
        "iterations": iterations,
        "elapsed_seconds": finished - started,
        "iterations_per_second": iterations / (finished - started),
        "thread_cpu": diff_thread_cpu(before, after),
    }


def profile_hyperloader_half(
    feeder: Any, workload: GpuWorkload, seconds: float, sampler: ClockSampler
) -> dict[str, object]:
    """Keep the hyperloader half identifiable in external stack samples."""
    return _profile_half(feeder, workload, seconds, sampler)


def profile_torch_half(
    feeder: Any, workload: GpuWorkload, seconds: float, sampler: ClockSampler
) -> dict[str, object]:
    """Keep the torch half identifiable in external stack samples."""
    return _profile_half(feeder, workload, seconds, sampler)


def _clock_windows(
    samples: list[dict[str, Any]], halves: dict[str, dict[str, object]]
) -> dict[str, object]:
    result = {}
    for name, half in halves.items():
        start = float(half["start_seconds"])
        end = float(half["end_seconds"])
        selected = [
            {**sample, "relative_seconds": float(sample["elapsed_seconds"]) - start}
            for sample in samples
            if start <= float(sample["elapsed_seconds"]) < end
        ]
        result[name] = summarize_clocks(selected, end - start)
    return result


def main() -> None:
    """Run equal-duration hyperloader and torch halves under continuous ALU activity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--half-seconds", type=float, default=20.0)
    parser.add_argument(
        "--system",
        choices=("hyperloader", "torch", "paired"),
        default="paired",
    )
    parser.add_argument("--profile-ready-file", type=Path)
    parser.add_argument("--profiler-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--spinner-library", type=Path, required=True)
    parser.add_argument("--spinner-cores", type=int, nargs="+", default=(17, 18))
    arguments = parser.parse_args()
    if arguments.half_seconds <= 0:
        raise ValueError("half duration must be positive")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    bundle = make_workload(
        "fixed-text", arguments.output.parent / "workload", batches=32
    )
    configs = {
        "hyperloader": SelectedConfig(workers=2, prefetch_factor=2),
        "torch": SelectedConfig(workers=8, prefetch_factor=4),
    }
    selected_names = (
        ("hyperloader", "torch")
        if arguments.system == "paired"
        else (arguments.system,)
    )
    feeders = {
        name: build_feeder(name, bundle, configs[name]) for name in selected_names
    }
    original_affinity = os.sched_getaffinity(0)
    sampler = ClockSampler(interval_seconds=1.0, affinity_cpu=14)
    spinner = AluSpinner(arguments.spinner_library, tuple(arguments.spinner_cores))
    sampler_started = False
    spinner_started = False
    try:
        for feeder in feeders.values():
            for _ in range(8):
                feeder.next_batch()
        warm_batch = feeders[selected_names[0]].next_batch()
        os.sched_setaffinity(0, {max(original_affinity)})
        workload = GpuWorkload("compute")
        workload.warm(warm_batch, iterations=20)
        if arguments.profile_ready_file is not None:
            profile_start = threading.Event()
            signal.signal(signal.SIGUSR1, lambda _signum, _frame: profile_start.set())
            arguments.profile_ready_file.parent.mkdir(parents=True, exist_ok=True)
            arguments.profile_ready_file.write_text(
                f"{os.getpid()}\n", encoding="utf-8"
            )
            if not profile_start.wait(arguments.profiler_timeout_seconds):
                raise TimeoutError("sampling profiler did not arm before the timeout")
        sampler.start()
        sampler_started = True
        spinner.start()
        spinner_started = True
        halves = {}
        if "hyperloader" in feeders:
            halves["hyperloader"] = profile_hyperloader_half(
                feeders["hyperloader"], workload, arguments.half_seconds, sampler
            )
        if "torch" in feeders:
            halves["torch"] = profile_torch_half(
                feeders["torch"], workload, arguments.half_seconds, sampler
            )
    finally:
        spinner_report = spinner.stop() if spinner_started else None
        clock_samples = sampler.stop() if sampler_started else []
        os.sched_setaffinity(0, original_affinity)
        for feeder in feeders.values():
            feeder.close()
        bundle.close()

    report = {
        "commit": arguments.commit,
        "configs": {name: asdict(config) for name, config in configs.items()},
        "system": arguments.system,
        "half_seconds": arguments.half_seconds,
        "cpu_governor": cpu_governor(),
        "hardware_monitors": hardware_monitor_facts(),
        "main_affinity_cpu": max(original_affinity),
        "clock_sampler_cpu": 14,
        "spinner": spinner_report,
        "halves": halves,
        "clock_samples": clock_samples,
        "clock_summaries": _clock_windows(clock_samples, halves),
    }
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["halves"], sort_keys=True))


if __name__ == "__main__":
    main()

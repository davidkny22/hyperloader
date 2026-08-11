"""Measure CPU-idle wake latency around paired fixed-text GPU waits."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from dominance_alu_spinner import DutyCycleAluSpinner, DutyCycleAluSpinnerGroup
from dominance_cpu_idle import (
    capture_kernel_counters,
    diff_cpuidle,
    diff_gpu_interrupts,
)
from dominance_feeders import build_feeder
from dominance_gpu_segments import quantile, summarize_clocks, summarize_segments
from dominance_protocol import SelectedConfig
from dominance_tensor_measure import collect_batches, describe_batch
from dominance_wait_workload import EventQueryGpuWorkload
from dominance_workloads import make_workload
from overhead_environment import (
    ClockSampler,
    cpu_governor,
    hardware_monitor_facts,
    platform_facts,
)
from overhead_workload import GpuWorkload


class WarmthController(Protocol):
    """Lifecycle shared by one-core and multi-core diagnostic pulses."""

    def start(self) -> None: ...

    def stop(self) -> dict[str, object]: ...


def _measure_half(
    name: str,
    workload: GpuWorkload,
    next_batch: Callable[[], Any],
    half_seconds: float,
    sampler: ClockSampler,
    warmth: WarmthController | None,
) -> dict[str, object]:
    before = capture_kernel_counters()
    warmth_report = None
    if warmth is not None:
        warmth.start()
    observations: list[dict[str, float]] = []
    started = sampler.elapsed_seconds()
    deadline = time.perf_counter() + half_seconds
    try:
        while time.perf_counter() < deadline:
            observations.append(workload.run_timed(next_batch()))
    finally:
        if warmth is not None:
            warmth_report = warmth.stop()
    finished = sampler.elapsed_seconds()
    after = capture_kernel_counters()
    workload_duration = finished - started
    cpuidle_duration = (
        int(dict(after["cpuidle"])["captured_monotonic_ns"])
        - int(dict(before["cpuidle"])["captured_monotonic_ns"])
    ) / 1_000_000_000.0
    summary = summarize_segments(observations)
    summary["cell_iterations_per_second"] = len(observations) / workload_duration
    query_counts = [
        item["event_queries"] for item in observations if "event_queries" in item
    ]
    if query_counts:
        summary["event_queries"] = _summarize_values(query_counts)
    return {
        "name": name,
        "start_seconds": started,
        "end_seconds": finished,
        "workload_duration_seconds": workload_duration,
        "observations": observations,
        "summary": summary,
        "kernel_counters_before": before,
        "kernel_counters_after": after,
        "cpuidle_delta": diff_cpuidle(
            dict(before["cpuidle"]), dict(after["cpuidle"]), cpuidle_duration
        ),
        "gpu_interrupt_delta": diff_gpu_interrupts(
            dict(before["gpu_interrupts"]), dict(after["gpu_interrupts"])
        ),
        "warmth": warmth_report,
    }


def _summarize_values(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "observations": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": quantile(ordered, 0.50),
        "p90": quantile(ordered, 0.90),
        "maximum": ordered[-1],
    }


def _clock_windows(
    samples: list[dict[str, Any]], halves: list[dict[str, object]]
) -> dict[str, object]:
    result = {}
    for half in halves:
        start = float(half["start_seconds"])
        end = float(half["end_seconds"])
        selected = [
            {**sample, "relative_seconds": float(sample["elapsed_seconds"]) - start}
            for sample in samples
            if start <= float(sample["elapsed_seconds"]) < end
        ]
        result[str(half["name"])] = summarize_clocks(selected, end - start)
    return result


def uses_live_hyperloader(mode: str) -> bool:
    """Return whether a diagnostic mode consumes the installed loader directly."""
    return mode in {"auto-controls", "targeted-warmth"}


def main() -> None:
    """Run one guarded fixed-text wait-attribution cell."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "blocking",
            "event-query",
            "consumer-warmth",
            "auto-controls",
            "targeted-warmth",
        ),
        required=True,
    )
    parser.add_argument("--half-seconds", type=float, default=45.0)
    parser.add_argument("--spinner-library", type=Path)
    parser.add_argument("--warmth-active-us", type=int, default=50)
    parser.add_argument("--warmth-period-us", type=int, default=1_000)
    parser.add_argument(
        "--hyper-delivery-memory",
        choices=("auto", "host", "pinned"),
        default="auto",
    )
    arguments = parser.parse_args()
    if arguments.half_seconds <= 0:
        raise ValueError("half duration must be positive")
    warmth_modes = {"consumer-warmth", "targeted-warmth"}
    if arguments.mode in warmth_modes and arguments.spinner_library is None:
        raise ValueError("warmth diagnostics require a native spinner library")
    if arguments.mode not in warmth_modes and arguments.spinner_library is not None:
        raise ValueError("only warmth diagnostics can load the native spinner")
    arguments.output.mkdir(parents=True, exist_ok=False)

    bundle = make_workload("fixed-text", arguments.output / "workload", batches=32)
    configs = {
        "hyperloader": SelectedConfig(workers=2, prefetch_factor=2),
        "torch": SelectedConfig(workers=8, prefetch_factor=4),
    }
    feeders = {
        "hyperloader": build_feeder(
            "hyperloader",
            bundle,
            configs["hyperloader"],
            delivery_memory=arguments.hyper_delivery_memory,
            machine_keeping=(
                "off" if arguments.mode == "targeted-warmth" else "auto"
            ),
        ),
        "torch": build_feeder("torch", bundle, configs["torch"]),
    }
    closed_bank = collect_batches(feeders["hyperloader"], 32)
    torch_bank = collect_batches(feeders["torch"], 32)
    for left, right in zip(closed_bank, torch_bank, strict=True):
        if not left.equal(right):
            raise RuntimeError("fixed-text feeder banks differ")
    for _ in range(8):
        feeders["torch"].next_batch()
    hyperloader_closed = False
    if not uses_live_hyperloader(arguments.mode):
        feeders["hyperloader"].close()
        hyperloader_closed = True
    bank_index = 0

    def next_bank() -> Any:
        nonlocal bank_index
        batch = closed_bank[bank_index % len(closed_bank)]
        bank_index += 1
        return batch

    original_affinity = os.sched_getaffinity(0)
    consumer_cpu = max(original_affinity)
    sampler = ClockSampler(interval_seconds=1.0, affinity_cpu=14)
    sampler_started = False
    try:
        os.sched_setaffinity(0, {consumer_cpu})
        workload: GpuWorkload = (
            EventQueryGpuWorkload("compute")
            if arguments.mode == "event-query"
            else GpuWorkload("compute")
        )
        workload.warm(closed_bank[0], iterations=20)
        sampler.start()
        sampler_started = True
        warmth = None
        if arguments.mode == "consumer-warmth":
            warmth = DutyCycleAluSpinner(
                arguments.spinner_library,
                consumer_cpu,
                active_microseconds=arguments.warmth_active_us,
                period_microseconds=arguments.warmth_period_us,
            )
        elif arguments.mode == "targeted-warmth":
            warmth = DutyCycleAluSpinnerGroup(
                arguments.spinner_library,
                (0, consumer_cpu),
                active_microseconds=arguments.warmth_active_us,
                period_microseconds=arguments.warmth_period_us,
            )
        if uses_live_hyperloader(arguments.mode):
            halves = [
                _measure_half(
                    "hyperloader",
                    workload,
                    feeders["hyperloader"].next_batch,
                    arguments.half_seconds,
                    sampler,
                    warmth,
                )
            ]
            hyperloader_report = feeders["hyperloader"].report()
            halves.append(
                _measure_half(
                    "torch",
                    workload,
                    feeders["torch"].next_batch,
                    arguments.half_seconds,
                    sampler,
                    None,
                )
            )
        else:
            hyperloader_report = None
            halves = [
                _measure_half(
                    "closed-bank",
                    workload,
                    next_bank,
                    arguments.half_seconds,
                    sampler,
                    warmth,
                ),
                _measure_half(
                    "torch",
                    workload,
                    feeders["torch"].next_batch,
                    arguments.half_seconds,
                    sampler,
                    None,
                ),
            ]
    finally:
        clock_samples = sampler.stop() if sampler_started else []
        os.sched_setaffinity(0, original_affinity)
        if not hyperloader_closed:
            feeders["hyperloader"].close()
        feeders["torch"].close()
        bundle.close()

    report = {
        "commit": arguments.commit,
        "mode": arguments.mode,
        "configs": {name: asdict(config) for name, config in configs.items()},
        "campaign_transfer": {
            "torch_pin_memory": False,
            "consumer_non_blocking": False,
        },
        "wait_mode": (
            "bounded CUDA event query"
            if arguments.mode == "event-query"
            else "blocking torch.cuda.synchronize"
        ),
        "cpu_governor": cpu_governor(),
        "platform": platform_facts(),
        "hardware_monitors": hardware_monitor_facts(),
        "consumer_cpu": consumer_cpu,
        "clock_sampler_cpu": 14,
        "batch_bank_size": len(closed_bank),
        "batch": describe_batch(closed_bank[0]),
        "hyperloader_report": hyperloader_report,
        "hyperloader_delivery_memory": arguments.hyper_delivery_memory,
        "product_machine_keeping": (
            "off" if arguments.mode == "targeted-warmth" else "auto"
        ),
        "halves": halves,
        "clock_samples": clock_samples,
        "clock_summaries": _clock_windows(clock_samples, halves),
    }
    path = arguments.output / "report.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({half["name"]: half["summary"] for half in halves}))


if __name__ == "__main__":
    main()

"""Attribute fixed-text host time across live, idle, and closed loader states."""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dominance_alu_spinner import AluSpinner
from dominance_feeders import build_feeder
from dominance_gpu_segments import quantile, summarize_clocks, summarize_segments
from dominance_protocol import SelectedConfig
from dominance_tensor_measure import collect_batches, describe_batch
from dominance_thread_activity import diff_thread_cpu, snapshot_threads
from dominance_workloads import make_workload
from overhead_environment import ClockSampler, cpu_governor, hardware_monitor_facts
from overhead_workload import GpuWorkload


def _measure_state(
    name: str,
    workload: GpuWorkload,
    next_batch: Callable[[], Any],
    seconds: float,
    sampler: ClockSampler,
    *,
    time_feeder: bool,
) -> dict[str, object]:
    observations: list[dict[str, float]] = []
    feeder_times = []
    thread_before = snapshot_threads()
    started = sampler.elapsed_seconds()
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        feeder_started = time.perf_counter()
        batch = next_batch()
        feeder_finished = time.perf_counter()
        observations.append(workload.run_timed(batch))
        if time_feeder:
            feeder_times.append(1000.0 * (feeder_finished - feeder_started))
    finished = sampler.elapsed_seconds()
    thread_after = snapshot_threads()
    summary = summarize_segments(observations)
    summary["cell_iterations_per_second"] = len(observations) / (finished - started)
    return {
        "name": name,
        "start_seconds": started,
        "end_seconds": finished,
        "duration_seconds": finished - started,
        "observations": observations,
        "summary": summary,
        "next_batch_ms": _summarize_values(feeder_times),
        "thread_cpu": diff_thread_cpu(thread_before, thread_after),
        "threads_before": list(thread_before.values()),
        "threads_after": list(thread_after.values()),
    }


def _summarize_values(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "observations": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": quantile(ordered, 0.50),
        "p90": quantile(ordered, 0.90),
        "p99": quantile(ordered, 0.99),
        "maximum": ordered[-1],
    }


def _profile_next_batch(
    feeder: Any, workload: GpuWorkload, seconds: float
) -> dict[str, object]:
    profile = cProfile.Profile()
    iterations = 0
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        profile.enable()
        batch = feeder.next_batch()
        profile.disable()
        workload.run(batch)
        iterations += 1
    rows = []
    for (filename, line, function), values in pstats.Stats(profile).stats.items():
        primitive_calls, total_calls, self_seconds, cumulative_seconds, _callers = (
            values
        )
        rows.append(
            {
                "filename": filename,
                "line": line,
                "function": function,
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "self_seconds": self_seconds,
                "cumulative_seconds": cumulative_seconds,
            }
        )
    rows.sort(key=lambda row: float(row["cumulative_seconds"]), reverse=True)
    return {"iterations": iterations, "top_cumulative": rows[:40]}


def _clock_windows(
    samples: list[dict[str, Any]], states: list[dict[str, object]]
) -> dict[str, object]:
    result = {}
    for state in states:
        start = float(state["start_seconds"])
        end = float(state["end_seconds"])
        selected = [
            {**sample, "relative_seconds": float(sample["elapsed_seconds"]) - start}
            for sample in samples
            if start <= float(sample["elapsed_seconds"]) < end
        ]
        result[str(state["name"])] = summarize_clocks(selected, end - start)
    return result


def main() -> None:
    """Run the three-state host-attribution cell under continuous ALU activity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--state-seconds", type=float, default=30.0)
    parser.add_argument("--profile-seconds", type=float, default=5.0)
    parser.add_argument("--spinner-library", type=Path, required=True)
    parser.add_argument("--spinner-cores", type=int, nargs="+", default=(17, 18))
    arguments = parser.parse_args()
    if arguments.state_seconds <= 0 or arguments.profile_seconds <= 0:
        raise ValueError("state and profile durations must be positive")
    arguments.output.mkdir(parents=True, exist_ok=False)

    bundle = make_workload("fixed-text", arguments.output / "workload", batches=32)
    config = SelectedConfig(workers=2, prefetch_factor=2)
    feeder = build_feeder("hyperloader", bundle, config)
    bank = collect_batches(feeder, 32)
    bank_index = 0

    def next_bank() -> Any:
        nonlocal bank_index
        value = bank[bank_index % len(bank)]
        bank_index += 1
        return value

    original_affinity = os.sched_getaffinity(0)
    closed = False
    sampler_started = False
    spinner_started = False
    sampler = ClockSampler(interval_seconds=1.0, affinity_cpu=14)
    spinner = AluSpinner(arguments.spinner_library, tuple(arguments.spinner_cores))
    try:
        for _ in range(8):
            feeder.next_batch()
        os.sched_setaffinity(0, {max(original_affinity)})
        workload = GpuWorkload("compute")
        workload.warm(bank[0], iterations=20)
        sampler.start()
        sampler_started = True
        spinner.start()
        spinner_started = True
        next_batch_profile = _profile_next_batch(
            feeder, workload, arguments.profile_seconds
        )
        states = [
            _measure_state(
                "live",
                workload,
                feeder.next_batch,
                arguments.state_seconds,
                sampler,
                time_feeder=True,
            ),
            _measure_state(
                "constructed-idle",
                workload,
                next_bank,
                arguments.state_seconds,
                sampler,
                time_feeder=False,
            ),
        ]
        close_started = time.perf_counter()
        feeder.close()
        close_seconds = time.perf_counter() - close_started
        closed = True
        states.append(
            _measure_state(
                "closed",
                workload,
                next_bank,
                arguments.state_seconds,
                sampler,
                time_feeder=False,
            )
        )
    finally:
        if not closed:
            feeder.close()
        spinner_report = spinner.stop() if spinner_started else None
        clock_samples = sampler.stop() if sampler_started else []
        os.sched_setaffinity(0, original_affinity)
        bundle.close()

    report = {
        "commit": arguments.commit,
        "config": asdict(config),
        "campaign_transfer": {
            "torch_pin_memory": False,
            "consumer_non_blocking": False,
        },
        "cpu_governor": cpu_governor(),
        "hardware_monitors": hardware_monitor_facts(),
        "main_affinity_cpu": max(original_affinity),
        "clock_sampler_cpu": 14,
        "spinner": spinner_report,
        "batch_bank_size": len(bank),
        "batch": describe_batch(bank[0]),
        "close_seconds": close_seconds,
        "next_batch_profile": next_batch_profile,
        "states": states,
        "clock_samples": clock_samples,
        "clock_summaries": _clock_windows(clock_samples, states),
    }
    path = arguments.output / "report.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({state["name"]: state["summary"] for state in states}))


if __name__ == "__main__":
    main()

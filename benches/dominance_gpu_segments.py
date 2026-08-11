"""Paired CUDA-segment and clock-residency measurement helpers."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

SEGMENT_KEYS = (
    "cuda_copy_ms",
    "cuda_kernel_ms",
    "cuda_total_ms",
    "host_copy_call_ms",
    "host_kernel_launch_ms",
    "host_sync_ms",
    "host_total_ms",
)


def run_segmented_pair(
    workload: Any,
    feeders: dict[str, Callable[[], Any]],
    order: tuple[str, str],
    half_seconds: float,
) -> dict[str, object]:
    """Run one uninterrupted alternating pair with raw per-iteration segments."""
    if half_seconds <= 0 or set(feeders) != set(order):
        raise ValueError(
            "a positive duration and exactly two ordered feeders are required"
        )
    observations = {name: [] for name in order}
    spills = 0
    started = time.perf_counter()
    midpoint = started + half_seconds
    finished = midpoint + half_seconds
    while (selected_at := time.perf_counter()) < finished:
        selected = order[0] if selected_at < midpoint else order[1]
        batch = feeders[selected]()
        segments = workload.run_timed(batch)
        completed = time.perf_counter()
        observed = order[0] if completed < midpoint else order[1]
        observations[observed].append(segments)
        if observed != selected:
            spills += 1
    summaries = {
        name: summarize_segments(values) for name, values in observations.items()
    }
    for name, values in observations.items():
        summaries[name]["cell_iterations_per_second"] = len(values) / half_seconds
    return {
        "order": list(order),
        "duration_seconds_per_half": half_seconds,
        "boundary_spill_operations": spills,
        "observations": observations,
        "summaries": summaries,
    }


def summarize_segments(observations: list[dict[str, float]]) -> dict[str, object]:
    """Summarize raw CUDA and host segments without discarding their observations."""
    if not observations:
        raise ValueError("segment summary requires observations")
    summary: dict[str, object] = {
        "iterations": len(observations),
    }
    for key in SEGMENT_KEYS:
        values = sorted(item[key] for item in observations)
        summary[key] = {
            "mean": statistics.fmean(values),
            "p10": quantile(values, 0.10),
            "p50": quantile(values, 0.50),
            "p90": quantile(values, 0.90),
        }
    host_total_seconds = sum(item["host_total_ms"] for item in observations) / 1000
    summary["gpu_operation_iterations_per_second"] = (
        len(observations) / host_total_seconds
    )
    return summary


def split_clock_samples(
    samples: list[dict[str, float | int]],
    order: tuple[str, str],
    half_seconds: float,
) -> dict[str, object]:
    """Assign one continuous clock trace to named halves and five-second buckets."""
    assigned: dict[str, list[dict[str, float | int]]] = {name: [] for name in order}
    for sample in samples:
        elapsed = float(sample["elapsed_seconds"])
        index = 0 if elapsed < half_seconds else 1
        relative = elapsed - index * half_seconds
        assigned[order[index]].append({**sample, "relative_seconds": relative})
    return {
        name: summarize_clocks(values, half_seconds)
        for name, values in assigned.items()
    }


def summarize_clocks(
    samples: list[dict[str, float | int]], half_seconds: float
) -> dict[str, object]:
    """Report frequency residency and time-bucket means for one named half."""
    if not samples:
        raise ValueError("clock summary requires samples")
    clocks = sorted(float(item["clock_mhz"]) for item in samples)
    count = len(samples)
    bands = {
        "at_least_2390_percent": 100 * sum(value >= 2390 for value in clocks) / count,
        "2370_to_2389_percent": 100
        * sum(2370 <= value < 2390 for value in clocks)
        / count,
        "2350_to_2369_percent": 100
        * sum(2350 <= value < 2370 for value in clocks)
        / count,
        "below_2350_percent": 100 * sum(value < 2350 for value in clocks) / count,
    }
    buckets = []
    start = 0.0
    while start < half_seconds:
        selected = [
            item
            for item in samples
            if start <= float(item["relative_seconds"]) < start + 5.0
        ]
        if selected:
            buckets.append(
                {
                    "start_seconds": start,
                    "end_seconds": min(start + 5.0, half_seconds),
                    "samples": len(selected),
                    "mean_clock_mhz": statistics.fmean(
                        float(item["clock_mhz"]) for item in selected
                    ),
                    "mean_utilization_percent": statistics.fmean(
                        float(item["utilization_percent"]) for item in selected
                    ),
                }
            )
        start += 5.0
    return {
        "samples": count,
        "mean_clock_mhz": statistics.fmean(clocks),
        "minimum_clock_mhz": clocks[0],
        "maximum_clock_mhz": clocks[-1],
        "p10_clock_mhz": quantile(clocks, 0.10),
        "p50_clock_mhz": quantile(clocks, 0.50),
        "p90_clock_mhz": quantile(clocks, 0.90),
        "mean_utilization_percent": statistics.fmean(
            float(item["utilization_percent"]) for item in samples
        ),
        "residency": bands,
        "time_buckets": buckets,
    }


def quantile(values: list[float], probability: float) -> float:
    """Interpolate one quantile from an ordered nonempty sequence."""
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction

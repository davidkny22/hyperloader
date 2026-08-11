"""Measure fixed-text GPU machine state under CPU activity and clock control."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dominance_alu_spinner import AluSpinner
from dominance_feeders import build_feeder
from dominance_gpu_segments import run_segmented_pair, split_clock_samples
from dominance_protocol import SelectedConfig
from dominance_tensor_measure import describe_batch
from dominance_workloads import make_workload
from overhead_environment import (
    ClockSampler,
    cpu_governor,
    hardware_monitor_facts,
    platform_facts,
)
from overhead_workload import GpuWorkload


def _measure_pair(
    workload: GpuWorkload,
    feeders: dict[str, Any],
    half_seconds: float,
    spinner: AluSpinner | None,
) -> dict[str, object]:
    order = ("hyperloader", "torch")
    sampler = ClockSampler(
        interval_seconds=min(1.0, half_seconds / 4.0), affinity_cpu=14
    )
    spinner_report: dict[str, object] | None = None

    def begin_half(name: str) -> None:
        nonlocal spinner_report
        if spinner is None:
            return
        if name == "hyperloader":
            spinner.start()
        else:
            spinner_report = spinner.stop()

    sampler.start()
    try:
        pair = run_segmented_pair(
            workload,
            {name: feeder.next_batch for name, feeder in feeders.items()},
            order,
            half_seconds,
            on_half_start=begin_half,
        )
    finally:
        clocks = sampler.stop()
        if spinner is not None and spinner_report is None:
            spinner_report = spinner.stop()
    pair["clock_samples"] = clocks
    pair["clock_summaries"] = split_clock_samples(clocks, order, half_seconds)
    pair["clock_sampler_cpu"] = 14
    pair["spark_hwmon_sources"] = sampler.rail_sources
    pair["spinner"] = spinner_report
    return pair


def main() -> None:
    """Run one live fixed-text pair in the selected machine-state mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--mode", choices=("spinner", "quiet"), required=True)
    parser.add_argument("--half-seconds", type=float, default=45.0)
    parser.add_argument("--spinner-library", type=Path)
    parser.add_argument("--spinner-cores", type=int, nargs="*", default=(17, 18))
    arguments = parser.parse_args()
    if arguments.half_seconds <= 0:
        raise ValueError("half duration must be positive")
    if arguments.mode == "spinner" and arguments.spinner_library is None:
        raise ValueError("spinner mode requires a native spinner library")
    if arguments.mode == "quiet" and arguments.spinner_library is not None:
        raise ValueError("quiet mode cannot load the native spinner")
    arguments.output.mkdir(parents=True, exist_ok=False)

    bundle = make_workload("fixed-text", arguments.output / "workload", batches=32)
    configs = {
        "hyperloader": SelectedConfig(workers=2, prefetch_factor=2),
        "torch": SelectedConfig(workers=8, prefetch_factor=4),
    }
    feeders = {
        name: build_feeder(name, bundle, selected) for name, selected in configs.items()
    }
    original_affinity = os.sched_getaffinity(0)
    try:
        for feeder in feeders.values():
            for _ in range(8):
                feeder.next_batch()
        warm_batch = feeders["hyperloader"].next_batch()
        os.sched_setaffinity(0, {max(original_affinity)})
        workload = GpuWorkload("compute")
        workload.warm(warm_batch, iterations=20)
        spinner = (
            AluSpinner(arguments.spinner_library, tuple(arguments.spinner_cores))
            if arguments.mode == "spinner"
            else None
        )
        pair = _measure_pair(workload, feeders, arguments.half_seconds, spinner)
    finally:
        os.sched_setaffinity(0, original_affinity)
        for feeder in feeders.values():
            feeder.close()
        bundle.close()

    report = {
        "commit": arguments.commit,
        "mode": arguments.mode,
        "configs": {name: asdict(config) for name, config in configs.items()},
        "campaign_transfer": {
            "torch_pin_memory": False,
            "consumer_non_blocking": False,
        },
        "cpu_governor": cpu_governor(),
        "platform": platform_facts(),
        "hardware_monitors": hardware_monitor_facts(),
        "main_affinity_cpu": max(original_affinity),
        "batch": describe_batch(warm_batch),
        "pair": pair,
    }
    path = arguments.output / "report.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(pair["summaries"], sort_keys=True))


if __name__ == "__main__":
    main()

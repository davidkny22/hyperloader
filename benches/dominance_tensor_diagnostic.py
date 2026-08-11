"""Isolate fixed-text delivery memory from loader availability on Spark."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from dominance_feeders import build_feeder, native_thread_affinity
from dominance_protocol import SelectedConfig
from dominance_tensor_measure import (
    build_variants,
    collect_batches,
    describe_batch,
    measure_live,
    measure_prefetched,
    measure_static,
)
from dominance_workloads import make_workload
from overhead_environment import ClockSampler, cpu_governor, platform_facts
from overhead_workload import GpuWorkload


def main() -> None:
    """Measure fixed-text storage variants under one warmed compute workload."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    arguments = parser.parse_args()
    if arguments.seconds <= 0:
        raise ValueError("seconds must be positive")
    arguments.output.mkdir(parents=True, exist_ok=False)

    workload_bundle = make_workload(
        "fixed-text", arguments.output / "workload", batches=32
    )
    configs = {
        "hyperloader": SelectedConfig(workers=2, prefetch_factor=2),
        "torch": SelectedConfig(workers=8, prefetch_factor=4),
    }
    feeders = {
        system: build_feeder(system, workload_bundle, selected)
        for system, selected in configs.items()
    }
    try:
        banks = {
            system: collect_batches(feeder, 32) for system, feeder in feeders.items()
        }
    finally:
        for feeder in feeders.values():
            feeder.close()
    for left, right in zip(banks["hyperloader"], banks["torch"], strict=True):
        if not left.equal(right):
            raise RuntimeError("fixed-text batch banks differ")
    variants = build_variants(banks["hyperloader"], banks["torch"])

    original_affinity = os.sched_getaffinity(0)
    sampler = ClockSampler()
    observations = []
    try:
        os.sched_setaffinity(0, {max(original_affinity)})
        gpu_workload = GpuWorkload("compute")
        for batches in variants.values():
            gpu_workload.warm(batches[0], iterations=4)
        sampler.start()
        names = tuple(variants)
        for round_index in range(4):
            order = names if round_index % 2 == 0 else tuple(reversed(names))
            for name in order:
                observations.append(
                    {
                        "round": round_index,
                        "variant": name,
                        **measure_static(
                            gpu_workload, variants[name], arguments.seconds
                        ),
                    }
                )
        clocks = sampler.stop()
    finally:
        if sampler._thread.is_alive():
            sampler.stop()
        os.sched_setaffinity(0, original_affinity)
        workload_bundle.close()

    static_means = {
        name: sum(
            item["iterations_per_second"]
            for item in observations
            if item["variant"] == name
        )
        / 4
        for name in variants
    }

    live_feeders = {
        system: build_feeder(system, workload_bundle, selected)
        for system, selected in configs.items()
    }
    live_observations = []
    live_sampler = ClockSampler()
    with native_thread_affinity():
        prefetch_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hyperloader-tensor-prefetch"
        )
        prefetch_executor.submit(lambda: None).result()
    try:
        for feeder in live_feeders.values():
            for _ in range(8):
                feeder.next_batch()
        os.sched_setaffinity(0, {max(original_affinity)})
        gpu_workload = GpuWorkload("compute")
        gpu_workload.warm(live_feeders["hyperloader"].next_batch(), iterations=20)
        live_sampler.start()
        names = (
            "hyper-live",
            "hyper-live-touch",
            "hyper-live-clone",
            "hyper-live-read-prefetched",
            "hyper-live-write-prefetched",
            "torch-live",
        )
        for round_index in range(4):
            order = names if round_index % 2 == 0 else tuple(reversed(names))
            for name in order:
                system = "torch" if name == "torch-live" else "hyperloader"
                live_observations.append(
                    {
                        "round": round_index,
                        "variant": name,
                        **(
                            measure_prefetched(
                                gpu_workload,
                                live_feeders[system],
                                prefetch_executor,
                                arguments.seconds,
                                writeback=name == "hyper-live-write-prefetched",
                            )
                            if name
                            in {
                                "hyper-live-read-prefetched",
                                "hyper-live-write-prefetched",
                            }
                            else measure_live(
                                gpu_workload,
                                live_feeders[system],
                                arguments.seconds,
                                clone=name == "hyper-live-clone",
                                touch=name == "hyper-live-touch",
                            )
                        ),
                    }
                )
        live_clocks = live_sampler.stop()
    finally:
        if live_sampler._thread.is_alive():
            live_sampler.stop()
        os.sched_setaffinity(0, original_affinity)
        prefetch_executor.shutdown(wait=True, cancel_futures=True)
        for feeder in live_feeders.values():
            feeder.close()
        workload_bundle.close()

    live_means = {
        name: {
            metric: sum(item[metric] for item in items) / len(items)
            for metric in items[0]
            if metric not in {"round", "variant"}
        }
        for name in (
            "hyper-live",
            "hyper-live-touch",
            "hyper-live-clone",
            "hyper-live-read-prefetched",
            "hyper-live-write-prefetched",
            "torch-live",
        )
        if (items := [item for item in live_observations if item["variant"] == name])
    }
    report = {
        "commit": arguments.commit,
        "configs": {name: asdict(config) for name, config in configs.items()},
        "cpu_governor": cpu_governor(),
        "platform": platform_facts(),
        "batch_bank_size": 32,
        "seconds_per_observation": arguments.seconds,
        "metadata": {
            name: describe_batch(batches[0]) for name, batches in variants.items()
        },
        "static_observations": observations,
        "static_mean_iterations_per_second": static_means,
        "static_clock_samples": clocks,
        "live_observations": live_observations,
        "live_means": live_means,
        "live_clock_samples": live_clocks,
    }
    path = arguments.output / "report.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

"""Isolate fixed-text delivery memory from loader availability on Spark."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dominance_feeders import build_feeder
from dominance_protocol import SelectedConfig
from dominance_workloads import make_workload
from overhead_environment import ClockSampler, cpu_governor, platform_facts
from overhead_workload import GpuWorkload


def describe_batch(batch: Any) -> dict[str, object]:
    """Describe layout and storage properties that can affect GPU consumption."""
    storage = batch.untyped_storage()
    return {
        "dtype": str(batch.dtype),
        "shape": list(batch.shape),
        "stride": list(batch.stride()),
        "storage_offset": int(batch.storage_offset()),
        "storage_bytes": int(storage.nbytes()),
        "logical_bytes": int(batch.numel() * batch.element_size()),
        "data_pointer_mod_4096": int(batch.data_ptr() % 4096),
        "contiguous": bool(batch.is_contiguous()),
        "pinned": bool(batch.is_pinned()),
        "shared": bool(batch.is_shared()),
    }


def build_variants(
    hyper_batches: list[Any], torch_batches: list[Any]
) -> dict[str, list[Any]]:
    """Build ownership variants once, outside every timed interval."""
    return {
        "hyper-view": hyper_batches,
        "hyper-clone": [batch.clone() for batch in hyper_batches],
        "hyper-shared-clone": [
            batch.clone().share_memory_() for batch in hyper_batches
        ],
        "torch-shared": torch_batches,
        "torch-clone": [batch.clone() for batch in torch_batches],
    }


def _collect_batches(feeder: Any, count: int) -> list[Any]:
    return [feeder.next_batch() for _ in range(count)]


def _measure(
    workload: GpuWorkload, batches: list[Any], seconds: float
) -> dict[str, float | int]:
    count = 0
    started = time.perf_counter()
    deadline = started + seconds
    while time.perf_counter() < deadline:
        workload.run(batches[count % len(batches)])
        count += 1
    elapsed = time.perf_counter() - started
    return {
        "iterations": count,
        "elapsed_seconds": elapsed,
        "iterations_per_second": count / elapsed,
    }


def _measure_live(
    workload: GpuWorkload,
    feeder: Any,
    seconds: float,
    *,
    clone: bool = False,
) -> dict[str, float | int]:
    count = 0
    feeder_seconds = 0.0
    clone_seconds = 0.0
    gpu_seconds = 0.0
    started = time.perf_counter()
    deadline = started + seconds
    while time.perf_counter() < deadline:
        before_feeder = time.perf_counter()
        batch = feeder.next_batch()
        after_feeder = time.perf_counter()
        if clone:
            batch = batch.clone()
        after_clone = time.perf_counter()
        workload.run(batch)
        completed = time.perf_counter()
        feeder_seconds += after_feeder - before_feeder
        clone_seconds += after_clone - after_feeder
        gpu_seconds += completed - after_clone
        count += 1
    elapsed = time.perf_counter() - started
    return {
        "iterations": count,
        "elapsed_seconds": elapsed,
        "iterations_per_second": count / elapsed,
        "feeder_seconds_per_iteration": feeder_seconds / count,
        "clone_seconds_per_iteration": clone_seconds / count,
        "gpu_seconds_per_iteration": gpu_seconds / count,
    }


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
            system: _collect_batches(feeder, 32) for system, feeder in feeders.items()
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
                        **_measure(gpu_workload, variants[name], arguments.seconds),
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
    try:
        for feeder in live_feeders.values():
            for _ in range(8):
                feeder.next_batch()
        os.sched_setaffinity(0, {max(original_affinity)})
        gpu_workload = GpuWorkload("compute")
        gpu_workload.warm(live_feeders["hyperloader"].next_batch(), iterations=20)
        live_sampler.start()
        names = ("hyper-live", "hyper-live-clone", "torch-live")
        for round_index in range(4):
            order = names if round_index % 2 == 0 else tuple(reversed(names))
            for name in order:
                system = "torch" if name == "torch-live" else "hyperloader"
                live_observations.append(
                    {
                        "round": round_index,
                        "variant": name,
                        **_measure_live(
                            gpu_workload,
                            live_feeders[system],
                            arguments.seconds,
                            clone=name == "hyper-live-clone",
                        ),
                    }
                )
        live_clocks = live_sampler.stop()
    finally:
        if live_sampler._thread.is_alive():
            live_sampler.stop()
        os.sched_setaffinity(0, original_affinity)
        for feeder in live_feeders.values():
            feeder.close()
        workload_bundle.close()

    live_means = {
        name: {
            metric: sum(
                item[metric] for item in live_observations if item["variant"] == name
            )
            / 4
            for metric in (
                "iterations_per_second",
                "feeder_seconds_per_iteration",
                "clone_seconds_per_iteration",
                "gpu_seconds_per_iteration",
            )
        }
        for name in ("hyper-live", "hyper-live-clone", "torch-live")
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

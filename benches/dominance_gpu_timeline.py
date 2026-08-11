"""Localize fixed-text GPU time in paired live and static cells."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dominance_feeders import build_feeder
from dominance_gpu_segments import run_segmented_pair, split_clock_samples
from dominance_protocol import SelectedConfig
from dominance_tensor_memory import build_pinned_clone_bank
from dominance_tensor_measure import collect_batches, describe_batch
from dominance_workloads import make_workload
from overhead_environment import ClockSampler, cpu_governor, platform_facts
from overhead_workload import GpuWorkload


def _measure_pair(
    workload: GpuWorkload,
    feeders: dict[str, Any],
    order: tuple[str, str],
    half_seconds: float,
) -> dict[str, object]:
    sampler = ClockSampler(interval_seconds=min(1.0, half_seconds / 4.0))
    sampler.start()
    try:
        pair = run_segmented_pair(workload, feeders, order, half_seconds)
        clocks = sampler.stop()
    finally:
        if sampler._thread.is_alive():
            sampler.stop()
    pair["clock_samples"] = clocks
    pair["clock_summaries"] = split_clock_samples(clocks, order, half_seconds)
    return pair


def main() -> None:
    """Run one live cell and one same-protocol static-memory cell."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--half-seconds", type=float, default=45.0)
    arguments = parser.parse_args()
    if arguments.half_seconds <= 0:
        raise ValueError("half duration must be positive")
    arguments.output.mkdir(parents=True, exist_ok=False)

    bundle = make_workload("fixed-text", arguments.output / "workload", batches=32)
    configs = {
        "hyperloader": SelectedConfig(workers=2, prefetch_factor=2),
        "torch": SelectedConfig(workers=8, prefetch_factor=4),
    }
    live_feeders = {
        name: build_feeder(name, bundle, selected) for name, selected in configs.items()
    }
    original_affinity = os.sched_getaffinity(0)
    try:
        banks = {
            name: collect_batches(feeder, 32) for name, feeder in live_feeders.items()
        }
        for left, right in zip(banks["hyperloader"], banks["torch"], strict=True):
            if not left.equal(right):
                raise RuntimeError("fixed-text live feeder banks differ")
        pinned = build_pinned_clone_bank(banks["hyperloader"])
        for feeder in live_feeders.values():
            for _ in range(8):
                feeder.next_batch()
        os.sched_setaffinity(0, {max(original_affinity)})
        live_workload = GpuWorkload("compute")
        live_workload.warm(banks["hyperloader"][0], iterations=20)
        live = _measure_pair(
            live_workload,
            {name: feeder.next_batch for name, feeder in live_feeders.items()},
            ("hyperloader", "torch"),
            arguments.half_seconds,
        )
    finally:
        os.sched_setaffinity(0, original_affinity)
        for feeder in live_feeders.values():
            feeder.close()

    os.sched_setaffinity(0, {max(original_affinity)})
    try:
        static_workload = GpuWorkload("compute")
        static_workload.warm(banks["hyperloader"][0], iterations=20)
        static_workload.warm(pinned[0], iterations=20)
        indices = {"pageable-identity": 0, "pinned-clone": 0}

        def next_static(name: str, values: list[Any]) -> Any:
            index = indices[name]
            indices[name] += 1
            return values[index % len(values)]

        static = _measure_pair(
            static_workload,
            {
                "pageable-identity": lambda: next_static(
                    "pageable-identity", banks["hyperloader"]
                ),
                "pinned-clone": lambda: next_static("pinned-clone", pinned),
            },
            ("pageable-identity", "pinned-clone"),
            arguments.half_seconds,
        )
    finally:
        os.sched_setaffinity(0, original_affinity)
        bundle.close()

    report = {
        "commit": arguments.commit,
        "configs": {name: asdict(config) for name, config in configs.items()},
        "campaign_transfer": {
            "torch_pin_memory": False,
            "consumer_non_blocking": False,
        },
        "cpu_governor": cpu_governor(),
        "platform": platform_facts(),
        "batch_bank_size": 32,
        "metadata": {
            "pageable-identity": describe_batch(banks["hyperloader"][0]),
            "pinned-clone": describe_batch(pinned[0]),
        },
        "live": live,
        "static": static,
    }
    path = arguments.output / "report.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "live": live["summaries"],
                "static": static["summaries"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

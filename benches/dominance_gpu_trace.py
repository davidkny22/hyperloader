"""Capture short NVTX-delimited fixed-text feeder traces."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from dominance_feeders import build_feeder
from dominance_protocol import SelectedConfig
from dominance_workloads import make_workload
from overhead_workload import GpuWorkload


def main() -> None:
    """Run equal iteration counts under both feeders in one NVTX trace window."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    arguments = parser.parse_args()
    if arguments.iterations <= 0:
        raise ValueError("iterations must be positive")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    bundle = make_workload(
        "fixed-text", arguments.output.parent / "trace-workload", batches=32
    )
    configs = {
        "hyperloader": SelectedConfig(workers=2, prefetch_factor=2),
        "torch": SelectedConfig(workers=8, prefetch_factor=4),
    }
    feeders = {
        name: build_feeder(name, bundle, selected) for name, selected in configs.items()
    }
    original_affinity = os.sched_getaffinity(0)
    measurements = {}
    try:
        first = {name: feeder.next_batch() for name, feeder in feeders.items()}
        if not first["hyperloader"].equal(first["torch"]):
            raise RuntimeError("fixed-text trace feeders differ")
        for feeder in feeders.values():
            for _ in range(8):
                feeder.next_batch()
        os.sched_setaffinity(0, {max(original_affinity)})
        workload = GpuWorkload("compute")
        workload.warm(first["hyperloader"], iterations=20)
        import torch

        torch.cuda.nvtx.range_push("trace-window")
        try:
            for name in ("hyperloader", "torch"):
                torch.cuda.nvtx.range_push(f"feeder:{name}")
                started = time.perf_counter()
                try:
                    for _ in range(arguments.iterations):
                        workload.run(feeders[name].next_batch())
                finally:
                    elapsed = time.perf_counter() - started
                    torch.cuda.nvtx.range_pop()
                measurements[name] = {
                    "iterations": arguments.iterations,
                    "elapsed_seconds": elapsed,
                    "iterations_per_second": arguments.iterations / elapsed,
                }
        finally:
            torch.cuda.nvtx.range_pop()
    finally:
        os.sched_setaffinity(0, original_affinity)
        for feeder in feeders.values():
            feeder.close()
        bundle.close()

    report = {
        "commit": arguments.commit,
        "configs": {name: asdict(config) for name, config in configs.items()},
        "order": ["hyperloader", "torch"],
        "measurements": measurements,
    }
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

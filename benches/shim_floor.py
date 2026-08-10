"""Measure exact Python-tier shim and RNG-context primitives on one pinned core."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import random
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hyperloader import _hyperloader, rng
from hyperloader.process.rng import WorkerRngContext
from hyperloader.process.worker import evaluate_sample
from hyperloader.rng import _user_code_context
from hyperloader.thread.pool import ThreadPool

METRICS = (
    "process_seed_no_draw",
    "process_seed_all",
    "process_shim_total",
    "thread_seed_no_draw",
    "thread_seed_all",
    "thread_shim_total",
)


class NoOpDataset:
    """Return one observable scalar with negligible user work."""

    def __getitem__(self, index: int) -> int:
        return index ^ 0x5A5A


@dataclass
class ShimOperations:
    """Retain exact tier machinery and its global-state restoration."""

    operations: dict[str, Callable[[int], int]]
    process_context: WorkerRngContext
    thread_pool: ThreadPool
    random_state: object
    numpy_state: tuple[Any, ...]
    torch_state: torch.Tensor

    def close(self) -> None:
        """Restore global RNG bindings and stop the retained thread executor."""
        self.thread_pool.close()
        self.process_context.clear()
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_state)


def build_operations() -> ShimOperations:
    """Bind every measured operation to the exact shipped tier implementation."""
    saved_random = random.getstate()
    saved_numpy = np.random.get_state()
    saved_torch = torch.get_rng_state()
    dataset = NoOpDataset()
    root_seed, epoch = 0x1234_5678_9ABC_DEF0, 17
    process_context = WorkerRngContext(0, 1)
    process_context.attach_dataset(dataset)
    thread_pool = ThreadPool(dataset, 1, root_seed, None, None)

    def process_seed_no_draw(position: int) -> int:
        return process_context.install(root_seed, epoch, position)

    def process_seed_all(position: int) -> int:
        process_context.install(root_seed, epoch, position)
        process_context._torch._ensure_armed()
        process_context._random.generator._ensure_armed()
        process_context._numpy._ensure_armed()
        sample = process_context.current_sample
        return sample[0] ^ sample[1]

    def process_shim_total(position: int) -> int:
        status, value = evaluate_sample(
            dataset,
            0,
            1,
            root_seed,
            epoch,
            position,
            position,
            process_context,
        )
        if status != 0:
            raise RuntimeError("process shim benchmark sample failed")
        return int(value)

    def thread_seed_no_draw(position: int) -> int:
        sample = _hyperloader._sample_rng_context(root_seed, epoch, position)
        with _user_code_context(sample):
            pass
        return sample[0] ^ sample[1]

    def thread_seed_all(position: int) -> int:
        sample = _hyperloader._sample_rng_context(root_seed, epoch, position)
        with _user_code_context(sample):
            torch_generator = rng()
            numpy_generator = rng("numpy")
            random_generator = rng("random")
        return (
            torch_generator.initial_seed()
            ^ int(numpy_generator.bit_generator.state["state"]["key"][0])
            ^ int(random_generator._key)
        )

    def thread_shim_total(position: int) -> int:
        value, cost_ns = thread_pool._evaluate(epoch, position, position)
        return int(value) ^ cost_ns

    return ShimOperations(
        operations={
            "process_seed_no_draw": process_seed_no_draw,
            "process_seed_all": process_seed_all,
            "process_shim_total": process_shim_total,
            "thread_seed_no_draw": thread_seed_no_draw,
            "thread_seed_all": thread_seed_all,
            "thread_shim_total": thread_shim_total,
        },
        process_context=process_context,
        thread_pool=thread_pool,
        random_state=saved_random,
        numpy_state=saved_numpy,
        torch_state=saved_torch,
    )


def measure_operations(
    iterations: int, warmup_iterations: int, trials: int
) -> list[tuple[str, int, int, int, float, int]]:
    """Return rotated per-trial timings with observable checksums."""
    if iterations <= 0 or warmup_iterations <= 0 or trials <= 0:
        raise ValueError("measurement counts must be positive")
    bound = build_operations()
    rows = []
    try:
        for operation in bound.operations.values():
            for position in range(warmup_iterations):
                operation(position)
        for trial in range(trials):
            order = METRICS[trial % len(METRICS) :] + METRICS[: trial % len(METRICS)]
            for metric in order:
                operation = bound.operations[metric]
                checksum = 0
                started = time.perf_counter_ns()
                for position in range(iterations):
                    checksum ^= operation(position)
                elapsed = time.perf_counter_ns() - started
                rows.append(
                    (metric, trial, iterations, elapsed, elapsed / iterations, checksum)
                )
    finally:
        bound.close()
    return rows


def pinned_core_metadata(core: int) -> dict[str, str]:
    """Validate one-core affinity and return its frequency controls."""
    if not hasattr(os, "sched_getaffinity") or os.sched_getaffinity(0) != {core}:
        raise RuntimeError(f"measurement process must be pinned only to core {core}")
    base = Path(f"/sys/devices/system/cpu/cpu{core}/cpufreq")
    governor = (base / "scaling_governor").read_text(encoding="utf-8").strip()
    if governor != "performance":
        raise RuntimeError("measurement core governor must be performance")
    return {
        "core": str(core),
        "governor": governor,
        "max_freq_khz": (base / "cpuinfo_max_freq").read_text(encoding="utf-8").strip(),
    }


def main() -> None:
    """Write one complete raw primitive report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--warmup-iterations", type=int, default=1_000)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.trials < 10:
        raise ValueError("measurement requires at least ten trials")
    metadata = {
        **pinned_core_metadata(arguments.core),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "iterations": str(arguments.iterations),
        "warmup_iterations": str(arguments.warmup_iterations),
        "trials": str(arguments.trials),
    }
    rows = measure_operations(
        arguments.iterations, arguments.warmup_iterations, arguments.trials
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for key, value in metadata.items():
            writer.writerow(("meta", key, value))
        writer.writerow(
            (
                "kind",
                "metric",
                "trial",
                "iterations",
                "elapsed_ns",
                "ns_per_op",
                "checksum",
            )
        )
        for row in rows:
            writer.writerow(("data", *row))


if __name__ == "__main__":
    main()

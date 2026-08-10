"""Measure exact per-sample RNG installation components on one pinned core."""

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
from torch.utils.data import get_worker_info

from hyperloader import _hyperloader
from hyperloader.process.rng import WorkerRngContext

METRICS = (
    "native_context",
    "coordinate_update",
    "torch_rekey",
    "random_rekey",
    "numpy_rekey",
    "worker_info_lazy",
    "full_install",
    "full_seeded_sample",
)


@dataclass
class InstallOperations:
    """Bound operations and state restoration for one measurement process."""

    context: WorkerRngContext
    operations: dict[str, Callable[[], Any]]
    random_state: object
    numpy_state: tuple[Any, ...]
    torch_state: torch.Tensor

    def close(self) -> None:
        """Restore process globals after the measurement window."""
        self.context.clear()
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_state)


def build_operations() -> InstallOperations:
    """Bind the exact installation path and each separately priced component."""
    saved_random = random.getstate()
    saved_numpy = np.random.get_state()
    saved_torch = torch.get_rng_state()
    context = WorkerRngContext(0, 1)
    context.attach_dataset((0,))
    root_seed, epoch, position = 0x1234_5678_9ABC_DEF0, 17, 29
    sample = _hyperloader._sample_rng_context(root_seed, epoch, position)
    torch_seed, key, _ = sample

    def native_context() -> Any:
        return _hyperloader._sample_rng_context(root_seed, epoch, position)

    def coordinate_update() -> Any:
        context._current.value = sample
        return sample

    def torch_rekey() -> None:
        context._current.value = _hyperloader._sample_rng_context(
            root_seed, epoch, position
        )
        context._torch._ensure_armed()

    def random_rekey() -> None:
        context._current.value = _hyperloader._sample_rng_context(
            root_seed, epoch, position
        )
        context._random.generator._ensure_armed()

    def numpy_rekey() -> None:
        context._current.value = _hyperloader._sample_rng_context(
            root_seed, epoch, position
        )
        context._numpy._ensure_armed()

    def worker_info_lazy() -> Any:
        if context._worker_info is None:
            raise RuntimeError("benchmark context has no worker identity")
        context._worker_info.begin_sample(torch_seed)
        return get_worker_info()

    def full_install() -> Any:
        return context.install(root_seed, epoch, position)

    def full_seeded_sample() -> Any:
        context.install(root_seed, epoch, position)
        context._torch._ensure_armed()
        context._random.generator._ensure_armed()
        context._numpy._ensure_armed()
        return context._current.value

    return InstallOperations(
        context,
        {
            "native_context": native_context,
            "coordinate_update": coordinate_update,
            "torch_rekey": torch_rekey,
            "random_rekey": random_rekey,
            "numpy_rekey": numpy_rekey,
            "worker_info_lazy": worker_info_lazy,
            "full_install": full_install,
            "full_seeded_sample": full_seeded_sample,
        },
        saved_random,
        saved_numpy,
        saved_torch,
    )


def measure_operations(
    iterations: int, warmup_iterations: int, trials: int
) -> list[tuple[str, int, int, int, float, int]]:
    """Return per-trial nanosecond measurements for every named operation."""
    if iterations <= 0 or warmup_iterations <= 0 or trials <= 0:
        raise ValueError("measurement counts must be positive")
    bound = build_operations()
    rows = []
    try:
        for operation in bound.operations.values():
            for _ in range(warmup_iterations):
                operation()
        for trial in range(trials):
            order = METRICS[trial % len(METRICS) :] + METRICS[: trial % len(METRICS)]
            for metric in order:
                operation = bound.operations[metric]
                started = time.perf_counter_ns()
                for _ in range(iterations):
                    operation()
                elapsed = time.perf_counter_ns() - started
                sample = bound.context._current.value
                checksum = 0 if sample is None else sample[0] ^ sample[1]
                checksum ^= int(bound.context._random.generator._key)
                checksum ^= int(bound.context._numpy._key[0])
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
    """Measure every component and write one machine-readable CSV report."""
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

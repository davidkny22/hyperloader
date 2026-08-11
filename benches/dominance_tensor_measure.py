"""Measurement helpers for fixed-text tensor delivery diagnostics."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

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


def collect_batches(feeder: Any, count: int) -> list[Any]:
    """Collect a bounded logical batch bank from one public feeder."""
    return [feeder.next_batch() for _ in range(count)]


def measure_static(
    workload: GpuWorkload, batches: list[Any], seconds: float
) -> dict[str, float | int]:
    """Measure GPU consumption over one retained batch bank."""
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


def measure_live(
    workload: GpuWorkload,
    feeder: Any,
    seconds: float,
    *,
    clone: bool = False,
    touch: bool = False,
) -> dict[str, float | int]:
    """Split synchronous feeder, touch, clone, and GPU time."""
    count = 0
    feeder_seconds = 0.0
    touch_seconds = 0.0
    clone_seconds = 0.0
    gpu_seconds = 0.0
    started = time.perf_counter()
    deadline = started + seconds
    while time.perf_counter() < deadline:
        before_feeder = time.perf_counter()
        batch = feeder.next_batch()
        after_feeder = time.perf_counter()
        if touch:
            int(batch.sum().item())
        after_touch = time.perf_counter()
        if clone:
            batch = batch.clone()
        after_clone = time.perf_counter()
        workload.run(batch)
        completed = time.perf_counter()
        feeder_seconds += after_feeder - before_feeder
        touch_seconds += after_touch - after_feeder
        clone_seconds += after_clone - after_touch
        gpu_seconds += completed - after_clone
        count += 1
    elapsed = time.perf_counter() - started
    return {
        "iterations": count,
        "elapsed_seconds": elapsed,
        "iterations_per_second": count / elapsed,
        "feeder_seconds_per_iteration": feeder_seconds / count,
        "touch_seconds_per_iteration": touch_seconds / count,
        "clone_seconds_per_iteration": clone_seconds / count,
        "gpu_seconds_per_iteration": gpu_seconds / count,
    }


def _next_touched(feeder: Any) -> Any:
    batch = feeder.next_batch()
    int(batch.sum().item())
    return batch


def measure_prefetched(
    workload: GpuWorkload,
    feeder: Any,
    executor: ThreadPoolExecutor,
    seconds: float,
) -> dict[str, float | int]:
    """Read the next identity batch while the GPU consumes the current batch."""
    count = 0
    wait_seconds = 0.0
    gpu_seconds = 0.0
    future = executor.submit(_next_touched, feeder)
    started = time.perf_counter()
    deadline = started + seconds
    while time.perf_counter() < deadline:
        before_wait = time.perf_counter()
        batch = future.result()
        after_wait = time.perf_counter()
        future = executor.submit(_next_touched, feeder)
        workload.run(batch)
        completed = time.perf_counter()
        wait_seconds += after_wait - before_wait
        gpu_seconds += completed - after_wait
        count += 1
    future.result()
    elapsed = time.perf_counter() - started
    return {
        "iterations": count,
        "elapsed_seconds": elapsed,
        "iterations_per_second": count / elapsed,
        "wait_seconds_per_iteration": wait_seconds / count,
        "gpu_seconds_per_iteration": gpu_seconds / count,
    }

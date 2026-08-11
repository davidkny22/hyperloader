"""Host-memory controls for fixed-text tensor diagnostics."""

from __future__ import annotations

import time
from typing import Any


class RegisteredHostStorages:
    """Register each distinct source storage with the CUDA host runtime."""

    def __init__(self, batches: list[Any]) -> None:
        storages = {
            (int(storage.data_ptr()), int(storage.nbytes()))
            for batch in batches
            if (storage := batch.untyped_storage()).nbytes()
        }
        self._storages = tuple(sorted(storages))
        self._registered: list[int] = []

    @property
    def storage_count(self) -> int:
        """Return the number of distinct registered storage allocations."""
        return len(self._storages)

    @property
    def total_bytes(self) -> int:
        """Return the total registered storage capacity."""
        return sum(size for _, size in self._storages)

    def __enter__(self) -> RegisteredHostStorages:
        import torch

        runtime = torch.cuda.cudart()
        try:
            for pointer, size in self._storages:
                result = runtime.cudaHostRegister(pointer, size, 0)
                if int(result) != 0:
                    raise RuntimeError(f"cudaHostRegister failed with {result}")
                self._registered.append(pointer)
        except BaseException:
            for pointer in reversed(self._registered):
                runtime.cudaHostUnregister(pointer)
            self._registered.clear()
            raise
        return self

    def __exit__(self, error_type: object, error: object, traceback: object) -> None:
        import torch

        runtime = torch.cuda.cudart()
        failure: object | None = None
        for pointer in reversed(self._registered):
            result = runtime.cudaHostUnregister(pointer)
            if int(result) != 0 and failure is None:
                failure = result
        self._registered.clear()
        if failure is not None and error_type is None:
            raise RuntimeError(f"cudaHostUnregister failed with {failure}")


def build_pinned_clone_bank(batches: list[Any]) -> list[Any]:
    """Copy a retained batch bank into CUDA-allocated pinned host tensors."""
    import torch

    pinned = [torch.empty_like(batch, pin_memory=True) for batch in batches]
    for target, source in zip(pinned, batches, strict=True):
        target.copy_(source)
    return pinned


def measure_writeback_traffic(batch: Any, iterations: int = 256) -> dict[str, object]:
    """Time in-place add stores against an equal-sized distinct-buffer copy."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    expected = batch.clone()
    copy_source = batch.clone()
    copy_target = batch.clone()
    for _ in range(8):
        batch.add_(0)
        copy_target.copy_(copy_source)

    empty_started = time.perf_counter()
    for _ in range(iterations):
        pass
    empty_seconds = time.perf_counter() - empty_started

    version_before = int(batch._version)
    add_started = time.perf_counter()
    for _ in range(iterations):
        batch.add_(0)
    add_seconds = time.perf_counter() - add_started
    version_delta = int(batch._version) - version_before

    copy_started = time.perf_counter()
    for _ in range(iterations):
        copy_target.copy_(copy_source)
    copy_seconds = time.perf_counter() - copy_started

    logical_bytes = int(batch.numel() * batch.element_size())
    traffic_bytes = 2 * logical_bytes * iterations
    return {
        "iterations": iterations,
        "logical_bytes_per_iteration": logical_bytes,
        "read_write_bytes_per_iteration": 2 * logical_bytes,
        "empty_seconds_per_iteration": empty_seconds / iterations,
        "add_seconds_per_iteration": add_seconds / iterations,
        "copy_seconds_per_iteration": copy_seconds / iterations,
        "add_read_write_gb_per_second": traffic_bytes / add_seconds / 1e9,
        "copy_read_write_gb_per_second": traffic_bytes / copy_seconds / 1e9,
        "version_delta": version_delta,
        "values_preserved": bool(batch.equal(expected)),
    }

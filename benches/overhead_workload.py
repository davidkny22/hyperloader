"""Continuous compute- and bandwidth-bound GPU work for paired cells."""

from __future__ import annotations

import time
from typing import Any


class GpuWorkload:
    """One warmed GPU operation consuming each delivered input batch."""

    def __init__(self, regime: str) -> None:
        import torch

        if regime not in {"compute", "bandwidth"}:
            raise ValueError("GPU regime must be compute or bandwidth")
        self._torch = torch
        self.regime = regime
        self._batch_checksum: Any = None
        if regime == "compute":
            self._left = torch.randn((4096, 4096), dtype=torch.float16, device="cuda")
            self._right = torch.randn((4096, 4096), dtype=torch.float16, device="cuda")
            self._output = torch.empty_like(self._left)
        else:
            self._source = torch.randn(64 * 1024 * 1024, device="cuda")
            self._output = torch.empty_like(self._source)
        self._segment_start = torch.cuda.Event(enable_timing=True)
        self._copy_end = torch.cuda.Event(enable_timing=True)
        self._kernel_end = torch.cuda.Event(enable_timing=True)

    def run(self, batch: Any) -> None:
        """Transfer one batch, execute one fixed GPU operation, and synchronize."""
        device_batch = batch.to("cuda", non_blocking=False)
        self._run_kernels(device_batch)
        self._torch.cuda.synchronize()

    def run_timed(self, batch: Any) -> dict[str, float]:
        """Split one iteration into CUDA copy, kernel, and host synchronization time."""
        host_started = time.perf_counter()
        self._segment_start.record()
        copy_started = time.perf_counter()
        device_batch = batch.to("cuda", non_blocking=False)
        copy_returned = time.perf_counter()
        self._copy_end.record()
        kernel_started = time.perf_counter()
        self._run_kernels(device_batch)
        kernel_returned = time.perf_counter()
        self._kernel_end.record()
        sync_started = time.perf_counter()
        self._torch.cuda.synchronize()
        completed = time.perf_counter()
        return {
            "cuda_copy_ms": self._segment_start.elapsed_time(self._copy_end),
            "cuda_kernel_ms": self._copy_end.elapsed_time(self._kernel_end),
            "cuda_total_ms": self._segment_start.elapsed_time(self._kernel_end),
            "host_copy_call_ms": 1000.0 * (copy_returned - copy_started),
            "host_kernel_launch_ms": 1000.0 * (kernel_returned - kernel_started),
            "host_sync_ms": 1000.0 * (completed - sync_started),
            "host_total_ms": 1000.0 * (completed - host_started),
        }

    def _run_kernels(self, device_batch: Any) -> None:
        self._batch_checksum = device_batch.sum()
        if self.regime == "compute":
            self._torch.mm(self._left, self._right, out=self._output)
        else:
            self._torch.mul(self._source, 1.0001, out=self._output)

    def warm(self, batch: Any, iterations: int = 20) -> None:
        """Warm kernels, allocations, and clocks outside timing."""
        for _ in range(iterations):
            self.run(batch)

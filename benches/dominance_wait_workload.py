"""GPU workload variant with bounded CUDA-event query waiting."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from overhead_workload import GpuWorkload


def bounded_event_query(
    event: Any,
    timeout_seconds: float,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> int:
    """Poll one CUDA event until completion or a fixed safety deadline."""
    if timeout_seconds <= 0:
        raise ValueError("event-query timeout must be positive")
    deadline = clock() + timeout_seconds
    queries = 0
    while True:
        queries += 1
        if event.query():
            return queries
        if clock() >= deadline:
            raise TimeoutError(
                f"CUDA event did not complete within {timeout_seconds:.3f} seconds"
            )


class EventQueryGpuWorkload(GpuWorkload):
    """Complete each GPU iteration through a bounded event-query spin."""

    def __init__(self, regime: str, *, timeout_seconds: float = 5.0) -> None:
        super().__init__(regime)
        if timeout_seconds <= 0:
            raise ValueError("event-query timeout must be positive")
        self._query_timeout_seconds = timeout_seconds

    def run(self, batch: Any) -> None:
        """Transfer, launch, and spin-query the completion event."""
        device_batch = batch.to("cuda", non_blocking=False)
        self._run_kernels(device_batch)
        self._kernel_end.record()
        try:
            bounded_event_query(self._kernel_end, self._query_timeout_seconds)
        except TimeoutError:
            self._torch.cuda.synchronize()
            raise

    def run_timed(self, batch: Any) -> dict[str, float]:
        """Split one iteration while polling its recorded completion event."""
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
        try:
            queries = bounded_event_query(
                self._kernel_end, self._query_timeout_seconds
            )
        except TimeoutError:
            self._torch.cuda.synchronize()
            raise
        completed = time.perf_counter()
        return {
            "cuda_copy_ms": self._segment_start.elapsed_time(self._copy_end),
            "cuda_kernel_ms": self._copy_end.elapsed_time(self._kernel_end),
            "cuda_total_ms": self._segment_start.elapsed_time(self._kernel_end),
            "host_copy_call_ms": 1000.0 * (copy_returned - copy_started),
            "host_kernel_launch_ms": 1000.0 * (kernel_returned - kernel_started),
            "host_sync_ms": 1000.0 * (completed - sync_started),
            "host_total_ms": 1000.0 * (completed - host_started),
            "event_queries": float(queries),
        }

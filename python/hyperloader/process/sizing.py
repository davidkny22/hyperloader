"""Fixed-frontier and transport-capacity resolution."""

from __future__ import annotations

from typing import Any

from ..config import AUTO


def frontier_depth(loader: Any) -> int:
    """Resolve the fixed frontier while preserving the two-batch liveness floor."""
    batch_size = loader.batch_size if loader.batch_size is not None else 1
    minimum = 2 * batch_size
    configured = loader.config.scheduler.frontier_depth
    return minimum if configured is AUTO else max(minimum, configured)


def queue_capacity(depth: int, worker_count: int) -> int:
    """Cover the frontier with power-of-two capacity on each worker transport."""
    per_worker = (depth + worker_count - 1) // worker_count
    return max(2, 1 << (per_worker - 1).bit_length())


def delivery_length(loader: Any) -> int:
    """Clip the scheduled position range when incomplete batches are dropped."""
    length = loader._plan.length
    if not loader.drop_last or loader.batch_size is None:
        return length
    return length - (length % loader.batch_size)

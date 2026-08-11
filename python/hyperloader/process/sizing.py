"""Fixed-frontier and transport-capacity resolution."""

from __future__ import annotations

import math
from typing import Any

from ..config import AUTO
from .resources import free_host_memory


def frontier_depth(loader: Any) -> int:
    """Resolve the active frontier from hints, profile statistics, and budgets."""
    minimum = frontier_minimum(loader)
    ceiling = frontier_ceiling(loader)
    configured = loader.config.scheduler.frontier_depth
    if configured is not AUTO:
        return _clip_budget(loader, min(ceiling, max(minimum, configured)))
    if loader.prefetch_factor is not AUTO and loader.prefetch_factor is not None:
        requested = loader.prefetch_factor * _worker_width(loader) * _batch_size(loader)
        return _clip_budget(loader, min(ceiling, max(minimum, requested)))
    ratio = _profile_ratio(loader)
    if ratio is None:
        return minimum
    requested = math.ceil(
        _worker_width(loader) * ratio * loader.config.factors.f_safety
    )
    return _clip_budget(loader, min(ceiling, max(minimum, requested)))


def frontier_ceiling(loader: Any) -> int:
    """Resolve the plan-frozen depth ceiling at the worker-width ceiling."""
    pool = getattr(loader, "_process_pool", None)
    if pool is not None:
        return pool.frontier_ceiling
    minimum = frontier_minimum(loader)
    ratio = _profile_ratio(loader)
    if ratio is None:
        ratio = loader.config.factors.f_var
    requested = math.ceil(
        _worker_width(loader) * ratio * loader.config.factors.f_safety
    )
    return _clip_budget(loader, max(minimum, requested))


def frontier_budget(loader: Any) -> int:
    """Resolve the host-memory budget for the per-rank speculative frontier."""
    configured = loader.config.scheduler.frontier_budget
    if configured is not AUTO:
        return configured
    return int(loader.config.factors.f_mem * free_host_memory())


def frontier_minimum(loader: Any) -> int:
    """Resolve the liveness floor in per-rank samples."""
    batch_size = _batch_size(loader)
    configured = loader.config.factors.d_min
    if configured is AUTO:
        return 2 * batch_size
    return max(batch_size, configured)


def queue_capacity(depth: int, worker_count: int) -> int:
    """Cover the frontier with power-of-two capacity on each worker transport."""
    per_worker = (depth + worker_count - 1) // worker_count
    return max(2, 1 << (per_worker - 1).bit_length())


def delivery_length(loader: Any) -> int:
    """Clip the scheduled position range when incomplete batches are dropped."""
    sampler_runtime = getattr(loader, "_sampler_runtime", None)
    if sampler_runtime is not None:
        return sampler_runtime.length
    length = loader._plan.length
    if not loader.drop_last or loader.batch_size is None:
        return length
    return length - (length % loader.batch_size)


def _batch_size(loader: Any) -> int:
    return loader.batch_size if loader.batch_size is not None else 1


def _worker_width(loader: Any) -> int:
    return loader.num_workers if isinstance(loader.num_workers, int) else 1


def _profile_ratio(loader: Any) -> float | None:
    profile = getattr(loader, "_cost_profile", None)
    if profile is None:
        return None
    statistics = profile.statistics()
    if statistics is None:
        return None
    mean_ns, p999_ns, _populated = statistics
    return max(1.0, p999_ns / mean_ns)


def _clip_budget(loader: Any, depth: int) -> int:
    pool = getattr(loader, "_process_pool", None)
    if pool is None:
        return depth
    budget = frontier_budget(loader)
    capacity = budget // pool.bytes_sample
    return max(frontier_minimum(loader), min(depth, capacity))

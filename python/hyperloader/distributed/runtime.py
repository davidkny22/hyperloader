"""Construction-time capture of the torch distributed topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hyperloader.config import AUTO


@dataclass(frozen=True, slots=True)
class CapturedTopology:
    """Record the rank view that a native map loader was built against."""

    rank: int
    world_size: int
    enabled: bool
    discovered: bool


def capture_topology(rank: Any, world_size: Any) -> CapturedTopology:
    """Resolve an explicit topology or snapshot an initialized process group."""
    if rank is AUTO and world_size is AUTO:
        runtime = _runtime_topology()
        if runtime is None:
            return CapturedTopology(0, 1, False, False)
        runtime_rank, runtime_world_size = runtime
        return CapturedTopology(runtime_rank, runtime_world_size, True, True)
    if not isinstance(rank, int) or not isinstance(world_size, int):
        raise TypeError(
            "distributed rank and world_size must both be explicit or both be auto"
        )
    return CapturedTopology(rank, world_size, True, False)


def validate_runtime_topology(topology: CapturedTopology) -> None:
    """Reject a process group whose topology differs from construction time."""
    runtime = _runtime_topology()
    if runtime is None:
        return
    runtime_rank, runtime_world_size = runtime
    if (runtime_rank, runtime_world_size) == (topology.rank, topology.world_size):
        return
    raise RuntimeError(
        "torch.distributed topology changed after DataLoader construction: "
        f"captured rank={topology.rank}, world_size={topology.world_size}; "
        f"current rank={runtime_rank}, world_size={runtime_world_size}. "
        "Construct a fresh DataLoader after process-group initialization."
    )


def _runtime_topology() -> tuple[int, int] | None:
    """Return the initialized torch topology without initializing it."""
    from torch import distributed

    if not distributed.is_available() or not distributed.is_initialized():
        return None
    return int(distributed.get_rank()), int(distributed.get_world_size())

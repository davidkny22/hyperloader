"""Optional stateful-source sharding hook."""

from __future__ import annotations

from typing import Any


def apply_source_shard(
    dataset: Any,
    topology: Any,
    lane: int,
    lane_count: int,
) -> Any:
    """Apply one lane's optional source shard contract before iteration."""
    shard = getattr(dataset, "shard", None)
    if shard is None:
        return dataset
    if not callable(shard):
        raise TypeError("iterable source shard must be callable")
    replacement = shard(topology.rank, topology.world_size, lane, lane_count)
    return dataset if replacement is None else replacement

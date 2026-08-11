"""Rank-local views of the native map sampler stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hyperloader.config import AUTO


@dataclass(frozen=True, slots=True)
class MapPlacement:
    """Map rank-local positions onto one topology-independent global stream."""

    dataset_length: int
    batch_size: int
    rank: int
    world_size: int
    drop_last: bool
    exact_count: bool
    enabled: bool

    @property
    def global_batch(self) -> int:
        """Return the topology-invariant global batch size."""
        return self.batch_size * self.world_size

    @property
    def length(self) -> int:
        """Return the number of positions delivered by this rank."""
        if not self.enabled:
            if self.drop_last:
                return self.dataset_length - self.dataset_length % self.batch_size
            return self.dataset_length
        full_batches, tail = divmod(self.dataset_length, self.global_batch)
        full_length = full_batches * self.batch_size
        if self.drop_last or tail == 0:
            return full_length
        if self.exact_count:
            start = self.rank * tail // self.world_size
            stop = (self.rank + 1) * tail // self.world_size
            return full_length + stop - start
        return (full_batches + 1) * self.batch_size

    @property
    def identity(self) -> bool:
        """Report whether local positions are unchanged dataset positions."""
        return not self.enabled or (
            self.world_size == 1 and self.length == self.dataset_length
        )

    @property
    def batch_transport_safe(self) -> bool:
        """Report whether every delivered batch is one contiguous dataset range."""
        return self.identity

    def coordinate(self, position: int) -> int:
        """Return the global sampler coordinate for one rank-local position."""
        if not 0 <= position < self.length:
            raise IndexError("rank-local sampler position is outside the placement")
        if not self.enabled:
            return position
        full_batches, tail = divmod(self.dataset_length, self.global_batch)
        full_length = full_batches * self.batch_size
        if position < full_length:
            batch, offset = divmod(position, self.batch_size)
            return batch * self.global_batch + self.rank * self.batch_size + offset
        if self.exact_count and tail:
            tail_start = self.rank * tail // self.world_size
            return (
                full_batches * self.global_batch + tail_start + position - full_length
            )
        batch, offset = divmod(position, self.batch_size)
        return batch * self.global_batch + self.rank * self.batch_size + offset

    def index(self, plan: Any, root_seed: int, epoch: int, position: int) -> int:
        """Return the dataset index assigned to one rank-local position."""
        coordinate = self.coordinate(position)
        source_position = (
            coordinate
            if coordinate < self.dataset_length
            else (coordinate - self.dataset_length) % self.dataset_length
        )
        return plan.index(root_seed, epoch, source_position)


def build_map_placement(loader: Any) -> MapPlacement:
    """Resolve explicit topology without consulting distributed runtime state."""
    length = loader._plan.length
    rank = loader.config.distributed.rank
    world_size = loader.config.distributed.world_size
    if rank is AUTO and world_size is AUTO:
        return MapPlacement(
            length,
            loader.batch_size or 1,
            0,
            1,
            loader.drop_last,
            loader.config.determinism.exact_count,
            False,
        )
    if not isinstance(rank, int) or not isinstance(world_size, int):
        raise TypeError(
            "distributed rank and world_size must both be explicit before runtime discovery"
        )
    return MapPlacement(
        length,
        loader.batch_size or 1,
        rank,
        world_size,
        loader.drop_last,
        loader.config.determinism.exact_count,
        True,
    )


def validate_elastic_restore(loader: Any, recorded_global_batch: int) -> None:
    """Require the current native topology to preserve a recorded global batch."""
    if (
        recorded_global_batch == 0
        or loader.sampler is not None
        or loader.batch_sampler is not None
    ):
        return
    placement = loader._map_placement
    world_size = placement.world_size
    if recorded_global_batch % world_size:
        quotient = f"{recorded_global_batch}/{world_size}"
        raise ValueError(
            f"recorded B_g={recorded_global_batch} cannot resume at world_size={world_size}; "
            f"the required per-rank batch_size={quotient} is not an integer"
        )
    required = recorded_global_batch // world_size
    actual = loader.batch_size or 1
    if actual != required:
        raise ValueError(
            f"recorded B_g={recorded_global_batch} at world_size={world_size} "
            f"requires per-rank batch_size={required}, not {actual}"
        )

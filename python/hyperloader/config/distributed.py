"""Explicit distributed topology configuration."""

from dataclasses import dataclass

from .automatic import AUTO, AutoInt, _require_nonnegative_int


@dataclass(frozen=True, slots=True)
class DistributedConfig:
    """Configure an explicit rank and world size when discovery is unavailable."""

    rank: AutoInt = AUTO
    world_size: AutoInt = AUTO

    def __post_init__(self) -> None:
        _require_nonnegative_int("distributed.rank", self.rank)
        _require_nonnegative_int("distributed.world_size", self.world_size)
        if self.world_size is not AUTO and self.world_size == 0:
            raise ValueError("distributed.world_size must be auto or positive")
        if (
            self.rank is not AUTO
            and self.world_size is not AUTO
            and self.rank >= self.world_size
        ):
            raise ValueError(
                "distributed.rank must be smaller than distributed.world_size"
            )

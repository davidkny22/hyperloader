"""Frontier and profile configuration."""

from dataclasses import dataclass
from os import PathLike

from .automatic import AUTO, AutoBytes, AutoInt, _require_nonnegative_int


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Configure frontier sizing, dispatch, and persisted cost profiles."""

    frontier_depth: AutoInt = AUTO
    frontier_budget: AutoBytes = AUTO
    early_dispatch: bool = True
    profile_cache: object = AUTO

    def __post_init__(self) -> None:
        _require_nonnegative_int("scheduler.frontier_depth", self.frontier_depth)
        _require_nonnegative_int("scheduler.frontier_budget", self.frontier_budget)
        if (
            self.profile_cache != "off"
            and not isinstance(self.profile_cache, (str, PathLike))
            and self.profile_cache is not AUTO
        ):
            raise ValueError("scheduler.profile_cache must be auto, a path, or off")

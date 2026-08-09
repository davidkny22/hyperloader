"""Process execution configuration."""

from dataclasses import dataclass
from typing import Literal

from .automatic import AUTO, AutoInt, _require_nonnegative_int


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
    """Configure process capacity, placement, and worker-death behavior."""

    process_ceiling: AutoInt = AUTO
    pin_policy: Literal["auto", "efficiency-first", "none"] = "auto"
    on_worker_death: Literal["close", "restart"] = "close"

    def __post_init__(self) -> None:
        _require_nonnegative_int("executor.process_ceiling", self.process_ceiling)

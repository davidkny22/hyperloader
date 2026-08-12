"""Private native-module surface implemented by pure Python."""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from . import profile as _profile
from . import rng as _rng
from . import schedule as _schedule
from . import telemetry as _telemetry
from . import transport as _transport

IS_FALLBACK = True
CostProfile = _profile.CostProfile
ProcessResources = _transport.ProcessResources
StaticSchedule = _schedule.StaticSchedule
Telemetry = _telemetry.Telemetry
WorkerCommand = _transport.WorkerCommand
WorkerEndpoint = _transport.WorkerEndpoint
feistel_permute = _rng.feistel_permute
materialized_permutation = _rng.materialized_permutation
permutation_index = _rng.permutation_index
rank_placements = _rng.rank_placements
rng_block = _rng.rng_block
rng_block_from_key = _rng.rng_block_from_key
sample_rng_context = _rng.sample_rng_context


def package_version() -> str:
    """Return installed package metadata without a native binary."""
    try:
        return version("hyperloader")
    except PackageNotFoundError:
        return "0.1.0"


def default_collate(batch: Any) -> Any:
    """Delegate exact collation semantics to the installed Torch anchor."""
    from torch.utils.data._utils.collate import default_collate as torch_collate

    return torch_collate(batch)


class MachineKeeper:
    """Parked fallback actuator for machines without native keep-warm support."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self._duty = 0.0

    def park(self) -> None:
        """Keep the fallback actuator parked."""

    def defer_park(self, _gap_ns: int) -> None:
        """Keep the fallback actuator parked after a gapless signal."""

    def duty(self) -> float:
        """Return zero native machine-keeping duty."""
        return self._duty

    def close(self) -> None:
        """Release no-op actuator state."""


def current_cpu() -> int | None:
    """Return the current CPU when the platform exposes it."""
    getter = getattr(os, "sched_getcpu", None)
    return None if getter is None else int(getter())

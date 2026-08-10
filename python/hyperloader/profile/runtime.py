"""Runtime construction for native bounded cost profiles."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .. import _hyperloader


def profile_budget_bytes(fraction: float, free_disk_bytes: int) -> int:
    """Apply the configured disk fraction without exceeding available bytes."""
    if not 0 < fraction:
        raise ValueError("profile disk fraction must be positive")
    if free_disk_bytes < 0:
        raise ValueError("free disk bytes must be nonnegative")
    return min(free_disk_bytes, int(fraction * free_disk_bytes))


def build_cost_profile(loader: Any) -> Any | None:
    """Create the plan-local profile under the configured disk-size clamp."""
    if loader.config.scheduler.profile_cache == "off" or loader._plan is None:
        return None
    free_disk = shutil.disk_usage(Path.cwd()).free
    max_bytes = profile_budget_bytes(loader.config.factors.f_prof, free_disk)
    return _hyperloader._CostProfile(
        loader._plan.length,
        max_bytes,
        loader.config.factors.alpha,
    )

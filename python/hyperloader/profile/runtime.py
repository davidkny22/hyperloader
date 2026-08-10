"""Runtime construction for native bounded cost profiles."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .. import _hyperloader
from ..config import AUTO
from ..control.cache import user_cache_root
from ..control.machine import MachineIdentity, detect_machine_identity
from ..fingerprint import ContractFingerprint


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
        loader._profile_cache_path = None
        return None
    machine = detect_machine_identity()
    loader._machine_identity = machine
    root = _profile_root(loader.config.scheduler.profile_cache)
    path = profile_cache_path(
        root,
        loader._dataset_fingerprint,
        machine,
    )
    free_disk = shutil.disk_usage(_existing_ancestor(root)).free
    max_bytes = profile_budget_bytes(loader.config.factors.f_prof, free_disk)
    loader._profile_cache_path = path
    if path.is_file():
        try:
            return _hyperloader._CostProfile.load(
                path,
                loader._plan.length,
                max_bytes,
                loader.config.factors.alpha,
            )
        except (OSError, ValueError):
            pass
    return _hyperloader._CostProfile(
        loader._plan.length, max_bytes, loader.config.factors.alpha
    )


def save_cost_profile(loader: Any) -> None:
    """Persist current estimates without making cache availability a contract input."""
    profile = getattr(loader, "_cost_profile", None)
    path = getattr(loader, "_profile_cache_path", None)
    if profile is None or path is None:
        return
    try:
        profile.save(path)
    except (OSError, ValueError):
        return


def profile_cache_path(
    root: Path, dataset: ContractFingerprint, machine: MachineIdentity
) -> Path:
    """Return the opaque dataset-and-machine profile cache path."""
    return root / dataset.digest / f"{machine.cache_key}.bin"


def _profile_root(configured: object) -> Path:
    if configured is AUTO:
        return user_cache_root() / "profiles"
    return Path(configured) / "profiles"


def _existing_ancestor(path: Path) -> Path:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise OSError(f"profile cache path has no existing ancestor: {path}")
        candidate = parent
    return candidate

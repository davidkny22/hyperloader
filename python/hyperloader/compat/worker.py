"""Worker initialization and free-running RNG state for compat lanes."""

from __future__ import annotations

import dataclasses
import pickle
from typing import Any

import torch

from .rng import capture_globals, restore_globals


class WorkerInitializer:
    """Restore torch-visible dataset identity and optional lane RNG state."""

    def __init__(
        self,
        user_init: Any,
        lane_states: dict[int, bytes],
        lane_seeds: dict[int, int],
    ) -> None:
        self.user_init = user_init
        self.lane_states = lane_states
        self.lane_seeds = lane_seeds

    def __call__(self, worker: int) -> None:
        patch_worker_dataset(self.lane_seeds.get(worker))
        if self.user_init is not None:
            self.user_init(worker)
        state = self.lane_states.get(worker)
        if state is not None:
            restore_worker_state(state)


def capture_worker_state() -> bytes:
    """Serialize the three free-running CPU RNG states in one lane."""
    return pickle.dumps(capture_globals(), protocol=pickle.HIGHEST_PROTOCOL)


def restore_worker_state(payload: bytes) -> None:
    """Restore one validated lane RNG state bundle."""
    restore_globals(pickle.loads(payload))


def patch_worker_dataset(seed: int | None = None) -> None:
    """Expose the user's dataset rather than the internal tagging adapter."""
    import torch.utils.data._utils.worker as worker_module

    info = torch.utils.data.get_worker_info()
    if info is None:
        raise RuntimeError("compat worker initializer ran outside a worker")
    adapter = info.dataset
    dataset = getattr(adapter, "dataset", adapter)
    if dataclasses.is_dataclass(info):
        names = [field.name for field in dataclasses.fields(info)]
    else:
        names = list(getattr(info, "_WorkerInfo__keys"))
    values = {}
    for name in names:
        value = getattr(info, name)
        if name == "dataset":
            value = dataset
        elif name == "seed" and seed is not None:
            value = seed
        values[name] = value
    worker_module._worker_info = type(info)(**values)

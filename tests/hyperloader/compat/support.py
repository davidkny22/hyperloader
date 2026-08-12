"""Shared fixtures for torch-compatible process-lane tests."""

from __future__ import annotations

import os
import random
from dataclasses import replace

import numpy as np
import torch
from hyperloader import HyperConfig
from hyperloader.config import DeterminismConfig
from torch.utils.data import get_worker_info


class LaneDataset:
    """Expose sample order, lane identity, and each worker RNG surface."""

    def __init__(self, length: int = 24) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[int, ...]:
        info = get_worker_info()
        if info is None:
            raise RuntimeError("lane dataset requires a worker process")
        return (
            index,
            info.id,
            info.num_workers,
            info.seed,
            int(info.dataset is self),
            random.getrandbits(32),
            int(np.random.randint(0, 1 << 31)),
            int(torch.randint(0, 1 << 31, ()).item()),
        )


class InitializedLaneDataset(LaneDataset):
    """Expose a mutation applied by the worker initializer."""

    def __init__(self, length: int = 12) -> None:
        super().__init__(length)
        self.worker_marker = -1

    def __getitem__(self, index: int) -> tuple[int, ...]:
        return (*super().__getitem__(index), self.worker_marker)


class FatalLaneDataset:
    """Terminate the worker assigned the first sample."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> int:
        if index == 0:
            os._exit(17)
        return index


class IterableLaneDataset(torch.utils.data.IterableDataset):
    """Provide a worker-sharded iterable for resume-boundary validation."""

    def __iter__(self):
        info = get_worker_info()
        worker = 0 if info is None else info.id
        workers = 1 if info is None else info.num_workers
        yield from range(worker, 8, workers)


def initialize_lane(worker: int) -> None:
    """Mutate the worker-owned dataset through torch's public identity view."""
    info = get_worker_info()
    if info is None:
        raise RuntimeError("worker initializer requires a worker process")
    info.dataset.worker_marker = 100 + worker


def generator(seed: int) -> torch.Generator:
    """Return one reproducibly seeded torch generator."""
    result = torch.Generator()
    result.manual_seed(seed)
    return result


def records(loader) -> list[tuple[int, ...]]:
    """Flatten collated columns into comparable sample records."""
    result = []
    for batch in loader:
        result.extend(
            tuple(int(value) for value in row)
            for row in zip(*(column.tolist() for column in batch), strict=True)
        )
    return result


def compat_config() -> HyperConfig:
    """Enable opt-in compatibility continuation."""
    return replace(
        HyperConfig(),
        determinism=DeterminismConfig(compat_resume="on"),
    )

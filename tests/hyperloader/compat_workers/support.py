"""Fixtures for the installed Torch worker-structure differential."""

from __future__ import annotations

import multiprocessing as mp
import random
import time

import torch
from torch.utils.data import get_worker_info


class WorkerViewDataset:
    """Expose worker topology and dataset-copy identity."""

    def __init__(self, length: int = 12) -> None:
        self.length = length
        self.owner_pid = mp.current_process().pid
        self.marker = -1

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[int, ...]:
        info = get_worker_info()
        if info is None:
            raise RuntimeError("worker view requires a process lane")
        return (
            index,
            info.id,
            info.num_workers,
            info.seed,
            int(info.dataset is self),
            int(mp.current_process().pid != self.owner_pid),
            self.marker,
            random.getrandbits(31),
        )


class CompletionOrderDataset(WorkerViewDataset):
    """Delay the first lane-zero batch so completion order is observable."""

    def __getitem__(self, index: int) -> tuple[int, ...]:
        if index == 0:
            time.sleep(0.05)
        return super().__getitem__(index)


class HungDataset:
    """Hold the first fetch beyond the configured timeout."""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> int:
        if index == 0:
            time.sleep(0.2)
        return index


class WorkerIterable(torch.utils.data.IterableDataset):
    """Expose logical worker sharding and uneven lane exhaustion."""

    def __iter__(self):
        info = get_worker_info()
        if info is None:
            raise RuntimeError("iterable worker view requires a process lane")
        count = 5 if info.id == 0 else 3
        for offset in range(count):
            yield info.id, info.num_workers, info.seed, offset


class FixedBatchSampler:
    """Yield one nonsequential batch partition."""

    def __iter__(self):
        yield [3, 1]
        yield [2, 0]

    def __len__(self) -> int:
        return 2


def initialize_worker(worker: int) -> None:
    """Mutate the worker-owned dataset after Torch seeding."""
    info = get_worker_info()
    if info is None:
        raise RuntimeError("initializer requires a worker process")
    info.dataset.marker = 100 + worker


def seeded_generator() -> torch.Generator:
    """Return the fixed generator used by every differential pair."""
    generator = torch.Generator()
    generator.manual_seed(1701)
    return generator


def collate_worker_rows(samples):
    """Preserve sample tuples under a user collate function."""
    return tuple(samples)


def records(loader) -> list[tuple[int, ...]]:
    """Flatten a default-collated worker stream into scalar tuples."""
    return [
        tuple(int(value) for value in row)
        for batch in loader
        for row in zip(*(column.tolist() for column in batch), strict=True)
    ]

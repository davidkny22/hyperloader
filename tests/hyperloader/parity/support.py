"""Importable datasets and scenario builders for cross-tier parity."""

from __future__ import annotations

import random
from collections import OrderedDict, namedtuple
from typing import Any

import numpy as np
import torch
from hyperloader import DataLoader, rng
from hyperloader.config import DistributedConfig, HyperConfig, SchedulerConfig
from hyperloader.stages import Collate, Source, ThreadSafety, Transform, pipeline
from torch.utils.data import get_worker_info

Pair = namedtuple("Pair", "left right")
CONFIG = HyperConfig(scheduler=SchedulerConfig(profile_cache="off"))


class NestedDataset:
    """Produce every common recursive leaf through seeded globals."""

    def __len__(self) -> int:
        return 9

    def __getitem__(self, index: int) -> OrderedDict[str, Any]:
        return OrderedDict(
            (
                ("index", index),
                ("text", f"row-{index}"),
                ("bytes", index.to_bytes(2, "little")),
                (
                    "pair",
                    Pair(
                        random.random(),
                        np.asarray(
                            [np.random.random(), np.random.randint(0, 1 << 20)],
                            dtype=np.float64,
                        ),
                    ),
                ),
                ("tensor", torch.rand(2)),
            )
        )


class AccessorDataset:
    """Use only the declared thread-safe generator surfaces."""

    def __len__(self) -> int:
        return 9

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "index": index,
            "numpy": np.asarray(rng("numpy").random(2), dtype=np.float64),
            "python": (rng("random").random(), rng("random").getrandbits(19)),
            "torch": torch.rand(2, generator=rng()),
        }


class AmbientDataset:
    """Violate the thread declaration through an ambient global draw."""

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int) -> tuple[int, float]:
        return index, random.random()


class ArrayDataset:
    """Produce homogeneous NumPy rows for batch-grain transport."""

    def __len__(self) -> int:
        return 10

    def __getitem__(self, index: int) -> np.ndarray:
        return np.asarray([index, index + 1, index + 2], dtype=np.int64)


class FixedSampler:
    """Yield one nonsequential sample order."""

    def __iter__(self):
        return iter((8, 2, 5, 0, 7, 1))

    def __len__(self) -> int:
        return 6


class FixedBatchSampler:
    """Yield variable nonsequential batches."""

    def __iter__(self):
        return iter(((4, 1, 8), (0,), (3, 7)))

    def __len__(self) -> int:
        return 3


class ParityIterable:
    """Produce logical-lane values with coordinate-bound seeded globals."""

    def __iter__(self):
        info = get_worker_info()
        lane = 0 if info is None else info.id
        lanes = 1 if info is None else info.num_workers
        for value in range(lane, 8, lanes):
            yield (
                lane,
                value,
                random.getrandbits(24),
                int(np.random.randint(0, 1 << 20)),
                int(torch.randint(0, 1 << 20, ()).item()),
            )


class FailingDataset:
    """Raise at one stable stream position."""

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int) -> int:
        if index == 4:
            raise KeyError("parity sentinel")
        return index


def increment(value: int) -> int:
    """Apply one stable pipeline transform."""
    return value + 1


def seeded_collate(values: list[Any]) -> dict[str, Any]:
    """Expose process-owned collation outputs and seeded globals."""
    return {
        "numpy": np.random.random(),
        "python": random.random(),
        "torch": torch.rand(2),
        "values": torch.utils.data.default_collate(values),
    }


def single_collate(value: Any) -> tuple[Any, float]:
    """Exercise user collation without automatic batching."""
    return value, random.random()


def failing_collate(values: list[Any]) -> Any:
    """Raise from the batch-level collation stage."""
    del values
    raise ValueError("collate parity sentinel")


def distributed_config(rank: int) -> HyperConfig:
    """Build one fixed distributed placement configuration."""
    return HyperConfig(
        distributed=DistributedConfig(world_size=3, rank=rank),
        scheduler=SchedulerConfig(profile_cache="off"),
    )


def build_pipeline() -> Any:
    """Build one isolated sample chain with engine default collation."""
    return pipeline(
        Source(tuple(range(9)), output_type=int),
        Transform(increment, input_type=int, output_type=int),
        Collate(
            torch.utils.data.default_collate,
            input_type=int,
            output_type=torch.Tensor,
            thread_safety=ThreadSafety.ISOLATED,
        ),
    )


def collect(loader: DataLoader) -> list[Any]:
    """Drain and close one public loader."""
    try:
        return list(loader)
    finally:
        loader.close()

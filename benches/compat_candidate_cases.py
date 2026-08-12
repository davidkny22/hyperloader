"""Compatibility oracle cases executed through hyperloader's public loader."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from hyperloader import DataLoader

from benches.compat_golden_cases import (
    FixedSampler,
    MapRngDataset,
    ShardedIterable,
    initialize_marker,
    random_collate,
)
from benches.compat_golden_model import encode_value


def generate_candidate_cases() -> dict[str, list[list[dict[str, Any]]]]:
    """Generate the full oracle matrix through the installed public import."""
    cases = {
        "zero_worker_shuffle": _zero_worker(),
        "worker_round_robin": _worker_round_robin(),
        "persistent_free_running": _persistent_workers(),
        "worker_initializer": _worker_initializer(),
        "sampler_and_user_collate": _sampler_and_collate(),
        "iterable_sharding": _iterable_sharding(),
    }
    return {
        name: [[encode_value(batch) for batch in epoch] for epoch in epochs]
        for name, epochs in cases.items()
    }


def _zero_worker() -> list[list[Any]]:
    _seed_globals(101)
    loader = _loader(
        MapRngDataset(12),
        batch_size=3,
        shuffle=True,
        num_workers=0,
        generator=_generator(41001),
    )
    try:
        return [list(loader)]
    finally:
        loader.close()


def _worker_round_robin() -> list[list[Any]]:
    loader = _loader(
        MapRngDataset(24),
        batch_size=3,
        shuffle=True,
        num_workers=2,
        generator=_generator(41002),
        persistent_workers=False,
        in_order=True,
    )
    try:
        return [list(loader)]
    finally:
        loader.close()


def _persistent_workers() -> list[list[Any]]:
    loader = _loader(
        MapRngDataset(12),
        batch_size=2,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        generator=_generator(41003),
        in_order=True,
    )
    try:
        return [list(loader), list(loader)]
    finally:
        loader.close()


def _worker_initializer() -> list[list[Any]]:
    loader = _loader(
        MapRngDataset(12, marker=True),
        batch_size=2,
        num_workers=2,
        worker_init_fn=initialize_marker,
        generator=_generator(41004),
        persistent_workers=False,
        in_order=True,
    )
    try:
        return [list(loader)]
    finally:
        loader.close()


def _sampler_and_collate() -> list[list[Any]]:
    loader = _loader(
        MapRngDataset(12),
        batch_size=4,
        sampler=FixedSampler(),
        num_workers=2,
        collate_fn=random_collate,
        generator=_generator(41005),
        persistent_workers=False,
        in_order=True,
    )
    try:
        return [list(loader)]
    finally:
        loader.close()


def _iterable_sharding() -> list[list[Any]]:
    loader = _loader(
        ShardedIterable(),
        batch_size=2,
        num_workers=2,
        generator=_generator(41006),
        persistent_workers=False,
        in_order=True,
    )
    try:
        return [list(loader)]
    finally:
        loader.close()


def _loader(dataset: Any, **options: Any) -> DataLoader:
    return DataLoader(dataset, mode="torch-compat", **options)


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _seed_globals(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

"""Deterministic Torch DataLoader cases that expose compatibility semantics."""

from __future__ import annotations

import inspect
import random
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from torch.utils.data import get_worker_info

from benches.compat_golden_model import encode_value


class MapRngDataset:
    """Expose sample order, lane identity, worker seed, and global RNG draws."""

    def __init__(self, length: int, *, marker: bool = False) -> None:
        self.length = length
        self.marker = marker
        self.worker_marker = -1

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[int, ...]:
        info = get_worker_info()
        return (
            index,
            -1 if info is None else info.id,
            -1 if info is None else info.seed,
            random.getrandbits(32),
            int(np.random.randint(0, 2**31)),
            int(torch.randint(0, 2**31, (1,)).item()),
            self.worker_marker if self.marker else -1,
        )


class ShardedIterable(torch.utils.data.IterableDataset):
    """Expose Torch's worker-local iterable sharding and RNG order."""

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        info = get_worker_info()
        worker = 0 if info is None else info.id
        workers = 1 if info is None else info.num_workers
        seed = -1 if info is None else info.seed
        for index in range(worker, 12, workers):
            yield (
                index,
                worker,
                seed,
                random.getrandbits(32),
                int(np.random.randint(0, 2**31)),
                int(torch.randint(0, 2**31, (1,)).item()),
            )


class FixedSampler:
    """Yield one pinned nonsequential order."""

    def __iter__(self) -> Iterator[int]:
        return iter((7, 1, 9, 0, 5, 3, 11, 2, 10, 4, 8, 6))

    def __len__(self) -> int:
        return 12


def initialize_marker(worker: int) -> None:
    """Record post-seeding initializer execution on the worker dataset copy."""
    info = get_worker_info()
    if info is None:
        raise RuntimeError("worker initializer requires a worker")
    info.dataset.worker_marker = 100 + worker


def random_collate(samples: list[tuple[int, ...]]) -> dict[str, Any]:
    """Expose user collation plus its position in each lane's RNG stream."""
    return {
        "samples": samples,
        "random": random.getrandbits(32),
        "numpy": int(np.random.randint(0, 2**31)),
        "torch": int(torch.randint(0, 2**31, (1,)).item()),
    }


def generate_cases() -> dict[str, list[list[dict[str, Any]]]]:
    """Generate every strict-order stream from real Torch loaders."""
    cases = {
        "zero_worker_shuffle": _zero_worker(),
        "worker_round_robin": _worker_round_robin(),
        "persistent_free_running": _persistent_workers(),
        "worker_initializer": _worker_initializer(),
        "sampler_and_user_collate": _sampler_and_collate(),
        "iterable_sharding": _iterable_sharding(),
    }
    return {name: [_encode_epoch(epoch) for epoch in epochs] for name, epochs in cases.items()}


def supports_in_order() -> bool:
    """Report whether the pinned Torch minor exposes strict-order selection."""
    return "in_order" in inspect.signature(torch.utils.data.DataLoader).parameters


def _zero_worker() -> list[list[Any]]:
    _seed_globals(101)
    loader = torch.utils.data.DataLoader(
        MapRngDataset(12),
        batch_size=3,
        shuffle=True,
        num_workers=0,
        generator=_generator(41001),
    )
    return [list(loader)]


def _worker_round_robin() -> list[list[Any]]:
    return [
        list(
            torch.utils.data.DataLoader(
                MapRngDataset(24),
                batch_size=3,
                shuffle=True,
                num_workers=2,
                generator=_generator(41002),
                **_order_option(),
            )
        )
    ]


def _persistent_workers() -> list[list[Any]]:
    loader = torch.utils.data.DataLoader(
        MapRngDataset(12),
        batch_size=2,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        generator=_generator(41003),
        **_order_option(),
    )
    return [list(loader), list(loader)]


def _worker_initializer() -> list[list[Any]]:
    loader = torch.utils.data.DataLoader(
        MapRngDataset(12, marker=True),
        batch_size=2,
        num_workers=2,
        worker_init_fn=initialize_marker,
        generator=_generator(41004),
        **_order_option(),
    )
    return [list(loader)]


def _sampler_and_collate() -> list[list[Any]]:
    loader = torch.utils.data.DataLoader(
        MapRngDataset(12),
        batch_size=4,
        sampler=FixedSampler(),
        num_workers=2,
        collate_fn=random_collate,
        generator=_generator(41005),
        **_order_option(),
    )
    return [list(loader)]


def _iterable_sharding() -> list[list[Any]]:
    loader = torch.utils.data.DataLoader(
        ShardedIterable(),
        batch_size=2,
        num_workers=2,
        generator=_generator(41006),
        **_order_option(),
    )
    return [list(loader)]


def _encode_epoch(epoch: list[Any]) -> list[dict[str, Any]]:
    return [encode_value(batch) for batch in epoch]


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _seed_globals(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _order_option() -> dict[str, bool]:
    return {"in_order": True} if supports_in_order() else {}

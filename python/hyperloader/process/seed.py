"""Construction-time root seed resolution."""

from __future__ import annotations

from typing import Any

MAX_U64 = (1 << 64) - 1


def resolve_root_seed(seed: int | None, generator: Any) -> int:
    """Resolve and validate the loader's recorded root seed."""
    if seed is None:
        if generator is not None:
            seed = int(generator.initial_seed())
        else:
            import torch

            seed = int(torch.empty((), dtype=torch.int64).random_().item())
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= MAX_U64:
        raise ValueError("seed must fit an unsigned 64-bit integer")
    return seed

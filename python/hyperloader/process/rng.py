"""Per-sample CPU RNG installation and torch worker identity."""

from __future__ import annotations

import random
from typing import Any

from hyperloader import _hyperloader


def install_sample_rng(root_seed: int, epoch: int, position: int) -> int:
    """Install the three seeded CPU globals and return the torch seed."""
    torch_seed, random_seed, numpy_words = _hyperloader._sample_seed_words(
        root_seed, epoch, position
    )
    import numpy as np
    import torch

    torch.default_generator.manual_seed(torch_seed)
    random.seed(random_seed)
    np.random.seed(np.asarray(numpy_words, dtype=np.uint32))
    return torch_seed


def set_worker_info(
    worker_id: int,
    worker_count: int,
    seed: int | None,
    dataset: Any,
) -> None:
    """Install torch's process-global worker view for the current sample."""
    from torch.utils.data._utils import worker as worker_module

    worker_module._worker_info = worker_module.WorkerInfo(
        id=worker_id,
        num_workers=worker_count,
        seed=seed,
        dataset=dataset,
    )


def clear_worker_info() -> None:
    """Release the worker-global dataset reference before process exit."""
    from torch.utils.data._utils import worker as worker_module

    worker_module._worker_info = None

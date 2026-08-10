"""Per-sample CPU RNG installation and torch worker identity."""

from __future__ import annotations

import random
from typing import Any

from hyperloader import _hyperloader


class WorkerRngContext:
    """Retain worker-local RNG and identity objects across sample installs."""

    def __init__(self, worker_id: int, worker_count: int, dataset: Any) -> None:
        import numpy as np
        import torch
        from torch.utils.data._utils import worker as worker_module

        self._torch_generator = torch.default_generator
        self._random_seed = random.seed
        self._numpy_seed = np.random.seed
        self._numpy_asarray = np.asarray
        self._numpy_uint32 = np.uint32
        self._worker_module = worker_module
        self._worker_info_type = worker_module.WorkerInfo
        self._worker_id = worker_id
        self._worker_count = worker_count
        self._dataset = dataset
        self._install_worker_info(None)

    def install(self, root_seed: int, epoch: int, position: int) -> int:
        """Install the three seeded CPU globals and the current worker seed."""
        torch_seed, random_seed, numpy_words = _hyperloader._sample_seed_words(
            root_seed, epoch, position
        )
        self._torch_generator.manual_seed(torch_seed)
        self._random_seed(random_seed)
        self._numpy_seed(self._numpy_asarray(numpy_words, dtype=self._numpy_uint32))
        self._install_worker_info(torch_seed)
        return torch_seed

    def _install_worker_info(self, seed: int | None) -> None:
        self._worker_module._worker_info = self._worker_info_type(
            id=self._worker_id,
            num_workers=self._worker_count,
            seed=seed,
            dataset=self._dataset,
        )

    def clear(self) -> None:
        """Release the worker-global dataset reference before process exit."""
        self._worker_module._worker_info = None

"""Per-sample CPU RNG installation."""

from __future__ import annotations

import random
import struct
from typing import Any

from hyperloader import _hyperloader

from .worker_info import WorkerInfoContext


class WorkerRngContext:
    """Retain worker-local RNG application objects across sample installs."""

    def __init__(self, worker_id: int, worker_count: int, dataset: Any) -> None:
        import numpy as np
        import torch

        self._torch_generator = torch.default_generator
        self._random_setstate = random.setstate
        self._random_unpack = struct.Struct("=625I").unpack
        self._numpy_setstate = np.random.set_state
        self._numpy_frombuffer = np.frombuffer
        self._numpy_uint32 = np.uint32
        self._worker_info = WorkerInfoContext(worker_id, worker_count, dataset)

    def install(self, root_seed: int, epoch: int, position: int) -> int:
        """Install the three seeded CPU globals and the current worker seed."""
        torch_seed, random_bytes, numpy_bytes = _hyperloader._sample_rng_states(
            root_seed, epoch, position
        )
        self._torch_generator.manual_seed(torch_seed)
        self._random_setstate((3, self._random_unpack(random_bytes), None))
        numpy_state = self._numpy_frombuffer(numpy_bytes, dtype=self._numpy_uint32)
        self._numpy_setstate(("MT19937", numpy_state, 624, 0, 0.0))
        self._worker_info.begin_sample(torch_seed)
        return torch_seed

    def clear(self) -> None:
        """Release the worker-global dataset reference before process exit."""
        self._worker_info.clear()

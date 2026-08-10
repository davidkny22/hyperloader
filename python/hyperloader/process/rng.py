"""Per-sample CPU RNG installation."""

from __future__ import annotations

from typing import Any

from hyperloader import _hyperloader

from .numpy_surface import NumpyModuleSurface
from .random_surface import RandomModuleSurface
from .worker_info import WorkerInfoContext


class WorkerRngContext:
    """Retain worker-local RNG application objects across sample installs."""

    def __init__(self, worker_id: int, worker_count: int) -> None:
        import torch

        self._torch_generator = torch.default_generator
        self._random = RandomModuleSurface()
        self._numpy = NumpyModuleSurface()
        self._worker_id = worker_id
        self._worker_count = worker_count
        self._worker_info: WorkerInfoContext | None = None

    def attach_dataset(self, dataset: Any) -> None:
        """Attach the dataset only after its payload has loaded."""
        if self._worker_info is not None:
            raise RuntimeError("worker RNG context already has a dataset")
        self._worker_info = WorkerInfoContext(
            self._worker_id, self._worker_count, dataset
        )

    def install(self, root_seed: int, epoch: int, position: int) -> int:
        """Install the three seeded CPU globals and the current worker seed."""
        torch_seed, key = _hyperloader._sample_rng_context(
            root_seed, epoch, position
        )
        self._random.rekey(key, position)
        self._numpy.rekey(key, position)
        self._torch_generator.manual_seed(torch_seed)
        if self._worker_info is None:
            raise RuntimeError("worker RNG context has no attached dataset")
        self._worker_info.begin_sample(torch_seed)
        return torch_seed

    def clear(self) -> None:
        """Release the worker-global dataset reference before process exit."""
        if self._worker_info is not None:
            self._worker_info.clear()
            self._worker_info = None
        self._numpy.clear()
        self._random.clear()

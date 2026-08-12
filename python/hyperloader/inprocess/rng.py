"""Batch-scoped CPU RNG installation for calling-process execution."""

from __future__ import annotations

import random

from hyperloader import _hyperloader
from hyperloader.process.numpy_surface import NumpyModuleSurface
from hyperloader.process.random_surface import RandomModuleSurface
from hyperloader.process.sample_rng import CurrentSample, SampleRng
from hyperloader.process.torch_surface import TorchModuleSurface


class InProcessRngSession:
    """Install sample streams while preserving the consumer's CPU RNG states."""

    def __init__(self) -> None:
        import numpy as np
        import torch

        self._np = np
        self._torch = torch
        self._random_state = random.getstate()
        self._numpy_state = np.random.get_state()
        self._torch_state = torch.default_generator.get_state()
        self._current = CurrentSample()
        self._random = RandomModuleSurface(self._current)
        self._numpy = NumpyModuleSurface(self._current)
        self._torch_surface = TorchModuleSurface(self._current)

    def install(self, root_seed: int, epoch: int, coordinate: int) -> SampleRng:
        """Select the exact stream for one map coordinate."""
        sample = _hyperloader._sample_rng_context(root_seed, epoch, coordinate)
        self._current.value = sample
        return sample

    def install_collate(self, root_seed: int, epoch: int, ordinal: int) -> SampleRng:
        """Select the batch-level collation stream."""
        words = _hyperloader._rng_block(root_seed, epoch, ordinal, 0, 1)
        _, key, coordinate = _hyperloader._sample_rng_context(root_seed, epoch, ordinal)
        sample = (words[0] | (words[1] << 32), key, coordinate)
        self._current.value = sample
        return sample

    def close(self) -> None:
        """Restore module bindings and the consumer's saved CPU RNG states."""
        self._torch_surface.clear()
        self._numpy.clear()
        self._random.clear()
        random.setstate(self._random_state)
        self._np.random.set_state(self._numpy_state)
        self._torch.default_generator.set_state(self._torch_state)

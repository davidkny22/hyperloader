"""Coordinate-bound RNG installation for sequential iterable lanes."""

from __future__ import annotations

import random
from typing import Any

from hyperloader import _hyperloader
from hyperloader.process.numpy_surface import NumpyModuleSurface
from hyperloader.process.random_surface import RandomModuleSurface
from hyperloader.process.sample_rng import CurrentSample, SampleRng
from hyperloader.process.torch_surface import TorchModuleSurface

_RANK_LIMIT = 1 << 12
_LANE_LIMIT = 1 << 12
_ARRIVAL_LIMIT = 1 << 40


def iterable_coordinate(rank: int, lane: int, arrival: int) -> int:
    """Pack one validated iterable sample identity into a stream coordinate."""
    _bounded("rank", rank, _RANK_LIMIT)
    _bounded("lane", lane, _LANE_LIMIT)
    _bounded("arrival", arrival, _ARRIVAL_LIMIT)
    return (rank << 52) | (lane << 40) | arrival


class IterableRngSession:
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

    def install(
        self,
        root_seed: int,
        epoch: int,
        rank: int,
        lane: int,
        arrival: int,
    ) -> SampleRng:
        """Select the exact stream for one lane arrival."""
        coordinate = iterable_coordinate(rank, lane, arrival)
        sample = _hyperloader._sample_rng_context(root_seed, epoch, coordinate)
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


def _bounded(name: str, value: Any, limit: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"iterable {name} must be an integer")
    if not 0 <= value < limit:
        raise ValueError(f"iterable {name} must be in [0, {limit})")

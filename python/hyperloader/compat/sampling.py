"""Sampler adapters for compat resume coordinates and lane phase."""

from __future__ import annotations

from typing import Any

from .protocol import TaggedIndex


class BatchSamplerAdapter:
    """Tag and skip whole sampler batches while preserving lazy iteration."""

    def __init__(
        self,
        batch_sampler: Any,
        *,
        skip: int = 0,
        phase: int = 0,
    ) -> None:
        self.batch_sampler = batch_sampler
        self.skip = skip
        self.phase = phase

    def __iter__(self):
        for dummy in range(self.phase):
            yield [TaggedIndex(-self.phase + dummy, 0, None, True)]
        iterator = iter(self.batch_sampler)
        for _ in range(self.skip):
            try:
                next(iterator)
            except StopIteration:
                return
        for batch, indices in enumerate(iterator, start=self.skip):
            yield [
                TaggedIndex(batch, offset, index)
                for offset, index in enumerate(indices)
            ]

    def __len__(self) -> int:
        return self.phase + max(0, len(self.batch_sampler) - self.skip)


class SamplerAdapter:
    """Tag and skip unbatched sampler positions."""

    def __init__(self, sampler: Any, *, skip: int = 0, phase: int = 0) -> None:
        self.sampler = sampler
        self.skip = skip
        self.phase = phase

    def __iter__(self):
        for dummy in range(self.phase):
            yield TaggedIndex(-self.phase + dummy, 0, None, True)
        iterator = iter(self.sampler)
        for _ in range(self.skip):
            try:
                next(iterator)
            except StopIteration:
                return
        for batch, index in enumerate(iterator, start=self.skip):
            yield TaggedIndex(batch, 0, index)

    def __len__(self) -> int:
        return self.phase + max(0, len(self.sampler) - self.skip)

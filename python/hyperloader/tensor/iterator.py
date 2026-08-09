"""Batch-view delivery for contiguous tensor datasets."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


class TensorIterator(Iterator[Any]):
    """Iterate a tensor plan without serialization or host materialization."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self._epoch = loader._epoch
        length = loader._plan.length
        if loader.drop_last and loader.batch_size is not None:
            length -= length % loader.batch_size
        self._length = length
        self._position = 0
        self._complete = False
        self._valid = True

    def __iter__(self) -> TensorIterator:
        return self

    def __next__(self) -> Any:
        if not self._valid:
            raise RuntimeError("tensor iterator is no longer active")
        if self._position >= self._length:
            self._finish_epoch()
            raise StopIteration
        batch_size = self._loader.batch_size
        if batch_size is None:
            index = self._loader._plan.index(
                self._loader.root_seed, self._epoch, self._position
            )
            self._position += 1
            return self._loader.dataset[index]
        start = self._position
        stop = min(start + batch_size, self._length)
        self._position = stop
        if not self._loader._plan.shuffle:
            return self._loader.dataset[start:stop]
        indices = [
            self._loader._plan.index(self._loader.root_seed, self._epoch, position)
            for position in range(start, stop)
        ]
        return self._loader.dataset[indices]

    def _finish_epoch(self) -> None:
        if not self._complete:
            self._loader._epoch += 1
            self._complete = True

    @property
    def complete(self) -> bool:
        """Report whether exhaustion advanced the loader epoch."""
        return self._complete

    def invalidate(self) -> None:
        """Prevent a replaced iterator from producing more batches."""
        self._valid = False

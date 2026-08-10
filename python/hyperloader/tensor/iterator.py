"""Batch-view delivery for contiguous tensor datasets."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from ..telemetry.delivery import build_delivery_telemetry


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
        self._delivery_telemetry = build_delivery_telemetry(loader)

    def __iter__(self) -> TensorIterator:
        return self

    def __next__(self) -> Any:
        started_ns = time.perf_counter_ns()
        previous_position = self._position
        value = self._next_value()
        if self._delivery_telemetry is not None:
            self._delivery_telemetry.record_delivery(
                self._position - previous_position,
                value.numel() * value.element_size(),
                started_ns,
            )
        return value

    def _next_value(self) -> Any:
        """Return the next storage-preserving tensor value."""
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
            sample = self._loader.dataset[index]
            self._loader._epoch_state.mark_delivered(self._epoch)
            return sample
        start = self._position
        stop = min(start + batch_size, self._length)
        self._position = stop
        if not self._loader._plan.shuffle:
            batch = self._loader.dataset[start:stop]
            self._loader._epoch_state.mark_delivered(self._epoch)
            return batch
        indices = [
            self._loader._plan.index(self._loader.root_seed, self._epoch, position)
            for position in range(start, stop)
        ]
        batch = self._loader.dataset[indices]
        self._loader._epoch_state.mark_delivered(self._epoch)
        return batch

    def _finish_epoch(self) -> None:
        if not self._complete:
            self._loader._epoch_state.complete(self._epoch)
            if self._delivery_telemetry is not None:
                self._delivery_telemetry.finish_epoch(self._epoch)
            self._complete = True

    @property
    def complete(self) -> bool:
        """Report whether exhaustion advanced the loader epoch."""
        return self._complete

    def invalidate(self) -> None:
        """Prevent a replaced iterator from producing more batches."""
        self._valid = False

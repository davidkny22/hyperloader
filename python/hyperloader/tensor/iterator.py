"""Batch-view delivery for contiguous tensor datasets."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from ..telemetry.delivery import DELIVERY_GROUP


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
        self._telemetry = loader._telemetry
        self._telemetry_flushed_position = 0
        self._telemetry_next_sample_position = 0
        sample = None if length == 0 else loader.dataset[0]
        self._telemetry_sample_bytes = (
            0 if sample is None else sample.numel() * sample.element_size()
        )

    def __iter__(self) -> TensorIterator:
        return self

    def __next__(self) -> Any:
        batch_size = self._loader.batch_size or 1
        sample_delivery = (
            self._telemetry is not None
            and self._position >= self._telemetry_next_sample_position
        )
        started_ns = time.perf_counter_ns() if sample_delivery else 0
        value = self._next_value()
        if sample_delivery:
            self._record_telemetry_sample(started_ns, batch_size)
        return value

    def _record_telemetry_sample(self, started_ns: int, batch_size: int) -> None:
        samples = self._position - self._telemetry_flushed_position
        self._telemetry.record_deliveries(
            samples,
            self._batch_ordinal(self._position)
            - self._batch_ordinal(self._telemetry_flushed_position),
            samples * self._telemetry_sample_bytes,
            time.perf_counter_ns() - started_ns,
        )
        self._telemetry_flushed_position = self._position
        self._telemetry_next_sample_position = self._position + batch_size * (
            DELIVERY_GROUP - 1
        )

    def _batch_ordinal(self, position: int) -> int:
        batch_size = self._loader.batch_size or 1
        return (position + batch_size - 1) // batch_size

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
            if self._loader._memory_ledger is not None:
                self._loader._memory_ledger.record(sample, 1)
            self._loader._epoch_state.mark_delivered(self._epoch)
            return sample
        start = self._position
        stop = min(start + batch_size, self._length)
        self._position = stop
        if not self._loader._plan.shuffle:
            batch = self._loader.dataset[start:stop]
            if self._loader._memory_ledger is not None:
                self._loader._memory_ledger.record(batch, stop - start)
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
            if self._telemetry is not None:
                self._flush_telemetry()
                self._telemetry.finish_epoch(self._epoch)
            self._complete = True

    def _flush_telemetry(self) -> None:
        if (
            self._telemetry is None
            or self._position == self._telemetry_flushed_position
        ):
            return
        samples = self._position - self._telemetry_flushed_position
        self._telemetry.record_counts(
            samples,
            self._batch_ordinal(self._position)
            - self._batch_ordinal(self._telemetry_flushed_position),
            samples * self._telemetry_sample_bytes,
        )
        self._telemetry_flushed_position = self._position

    @property
    def complete(self) -> bool:
        """Report whether exhaustion advanced the loader epoch."""
        return self._complete

    def invalidate(self) -> None:
        """Prevent a replaced iterator from producing more batches."""
        self._valid = False

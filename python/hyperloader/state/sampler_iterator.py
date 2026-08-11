"""Strict batch delivery for user batch-sampler streams."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..process.factory import prepare_process_pool


class UserBatchSamplerIterator(Iterator[Any]):
    """Execute exact user batches without native placement or regrouping."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self._runtime = loader._sampler_runtime
        self._epoch = loader._epoch
        self._ordinal = self._runtime.start_batch
        self._complete = False
        self._valid = True
        if self._ordinal < self._runtime.length:
            prepare_process_pool(loader)

    def __iter__(self) -> UserBatchSamplerIterator:
        return self

    def __next__(self) -> Any:
        if not self._valid:
            raise RuntimeError("batch-sampler iterator is no longer active")
        if self._ordinal >= self._runtime.length:
            self._finish_epoch()
            raise StopIteration
        batch = self._runtime.batch(self._ordinal)
        sample_offset = self._runtime.sample_offset(self._ordinal)
        try:
            values = [
                self._loader._process_pool.execute(
                    self._epoch, sample_offset + offset, index
                )
                for offset, index in enumerate(batch)
            ]
            value = self._loader._collate_batch(values)
        except BaseException:
            self._loader.close()
            raise
        self._ordinal += 1
        self._loader._epoch_state.mark_delivered(self._epoch)
        return value

    @property
    def complete(self) -> bool:
        """Report whether exhaustion advanced the loader epoch."""
        return self._complete

    @property
    def coordinate_epoch(self) -> int:
        """Return the epoch carried by this iterator's checkpoint coordinate."""
        return self._epoch

    @property
    def delivered_batches(self) -> int:
        """Return the exact user-batch prefix count."""
        return self._ordinal

    @property
    def sampler_checksum(self) -> int:
        """Return the checksum through the delivered user-batch prefix."""
        return self._runtime.checksum_at(self._ordinal)

    def invalidate(self) -> None:
        """Prevent a replaced iterator from delivering more user batches."""
        self._valid = False

    def _finish_epoch(self) -> None:
        if self._complete:
            return
        self._loader._epoch_state.complete(self._epoch)
        self._complete = True

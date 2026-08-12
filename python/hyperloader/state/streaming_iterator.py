"""Sequential engine execution for unsized user sampler streams."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..process.factory import prepare_process_pool


class StreamingSamplerIterator(Iterator[Any]):
    """Deliver an unsized sampler without native placement or materialization."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self._runtime = loader._sampler_runtime
        self._epoch = loader._epoch
        self._position = self._runtime.start_samples
        self._delivered = loader._resume_cursor_batches
        self._complete = False
        self._valid = True
        if self._runtime.probe() is not None:
            prepare_process_pool(loader)

    def __iter__(self) -> StreamingSamplerIterator:
        return self

    def __next__(self) -> Any:
        if not self._valid:
            raise RuntimeError("streaming sampler iterator is no longer active")
        batch_size = self._loader.batch_size
        width = batch_size or 1
        indices = []
        while len(indices) < width and self._runtime.has_index(
            self._position + len(indices)
        ):
            indices.append(self._runtime.index(self._position + len(indices)))
        if not indices or (
            len(indices) < width and batch_size is not None and self._loader.drop_last
        ):
            self._finish_epoch()
            raise StopIteration
        try:
            if self._loader.collate_fn is not None:
                value = self._loader._process_pool.execute_collated(
                    self._epoch,
                    self._delivered,
                    tuple(
                        (self._position + offset, index)
                        for offset, index in enumerate(indices)
                    ),
                    auto_collation=batch_size is not None,
                )
            else:
                values = [
                    self._loader._process_pool.execute(
                        self._epoch, self._position + offset, index
                    )
                    for offset, index in enumerate(indices)
                ]
                value = (
                    values[0]
                    if batch_size is None
                    else self._loader._collate_batch(values)
                )
        except BaseException:
            self._loader.close()
            raise
        self._position += len(indices)
        self._delivered += 1
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
        """Return the delivered item or automatic-batch prefix."""
        return self._delivered

    @property
    def sampler_checksum(self) -> int:
        """Return the checksum through the delivered sample prefix."""
        return self._runtime.checksum_at(self._position)

    def invalidate(self) -> None:
        """Prevent a replaced iterator from delivering more sampler values."""
        self._valid = False

    def _finish_epoch(self) -> None:
        if self._complete:
            return
        self._loader._epoch_state.complete(self._epoch)
        self._complete = True

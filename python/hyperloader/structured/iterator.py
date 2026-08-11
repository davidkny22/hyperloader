"""Strict-order delivery from batch-native structured adapters."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..telemetry.delivery import build_delivery_telemetry
from ..process.sizing import frontier_depth
from ..state import resume_sample_position
from .metrics import payload_bytes


class StructuredIterator(Iterator[Any]):
    """Deliver one adapter-produced batch without per-sample transport."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self._epoch = loader._epoch
        length = loader._plan.length
        if loader.drop_last and loader.batch_size is not None:
            length -= length % loader.batch_size
        self._length = length
        self._position = resume_sample_position(loader, length)
        self._complete = False
        self._valid = True
        self._delivery_telemetry = build_delivery_telemetry(loader)
        begin_epoch = getattr(loader._execution_dataset, "begin_native_epoch", None)
        if begin_epoch is not None:
            batch_size = loader.batch_size or 1
            retained_stop = self._position
            if self._position == 0 and loader._native_batch_probe is not None:
                retained_stop = min(batch_size, length)
            elif self._position:
                loader._native_batch_probe = None
            begin_epoch(length, frontier_depth(loader), retained_stop, batch_size)

    def __iter__(self) -> StructuredIterator:
        return self

    def __next__(self) -> Any:
        if not self._valid:
            raise RuntimeError("structured iterator is no longer active")
        if self._position >= self._length:
            self._finish_epoch()
            raise StopIteration
        batch_size = self._loader.batch_size
        if batch_size is None:
            raise RuntimeError("batch-native delivery requires automatic batching")
        start = self._position
        stop = min(start + batch_size, self._length)
        self._position = stop
        started_ns = (
            0
            if self._delivery_telemetry is None
            else self._delivery_telemetry.start_delivery()
        )
        try:
            value = self._next_batch(start, stop)
        except BaseException:
            self._position = start
            self._loader.close()
            raise
        self._loader._epoch_state.mark_delivered(self._epoch)
        if self._delivery_telemetry is not None:
            self._delivery_telemetry.record_delivery(
                stop - start, payload_bytes(value), started_ns
            )
        return value

    def _next_batch(self, start: int, stop: int) -> Any:
        probe = self._loader._native_batch_probe
        if start != 0 or probe is None:
            return self._loader._execution_dataset.native_batch(start, stop)
        self._loader._native_batch_probe = None
        if probe.error is not None:
            raise probe.error
        return probe.value

    def _finish_epoch(self) -> None:
        if self._complete:
            return
        self._loader._epoch_state.complete(self._epoch)
        if self._delivery_telemetry is not None:
            self._delivery_telemetry.finish_epoch(self._epoch)
        self._complete = True

    def _flush_telemetry(self) -> None:
        if self._delivery_telemetry is not None:
            self._delivery_telemetry.flush()

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
        """Return the strict delivered-batch prefix count."""
        batch_size = self._loader.batch_size or 1
        return (self._position + batch_size - 1) // batch_size

    def invalidate(self) -> None:
        """Prevent a replaced iterator from producing more batches."""
        self._valid = False

"""Iterator wrapper for staged pinned delivery."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


class PinnedDeliveryIterator(Iterator[Any]):
    """Apply the loader-owned staging pool to each delivered value."""

    def __init__(self, delivery: Any, iterator: Iterator[Any]) -> None:
        self._delivery = delivery
        self._iterator = iterator

    def __iter__(self) -> PinnedDeliveryIterator:
        return self

    def __next__(self) -> Any:
        return self._delivery.stage(next(self._iterator))

    def _flush_telemetry(self) -> None:
        flush = getattr(self._iterator, "_flush_telemetry", None)
        if flush is not None:
            flush()

    @property
    def complete(self) -> bool:
        """Report the wrapped iterator's completion state."""
        return bool(self._iterator.complete)

    @property
    def coordinate_epoch(self) -> int:
        """Return the wrapped iterator's checkpoint epoch."""
        return int(self._iterator.coordinate_epoch)

    @property
    def delivered_batches(self) -> int:
        """Return the wrapped iterator's delivered-batch prefix."""
        return int(self._iterator.delivered_batches)

    def invalidate(self) -> None:
        """Invalidate the wrapped execution iterator."""
        self._iterator.invalidate()

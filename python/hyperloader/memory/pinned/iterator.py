"""Iterator wrapper for staged pinned delivery."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from queue import Queue
from threading import Event, Thread, current_thread, get_ident
from typing import Any


@dataclass(frozen=True, slots=True)
class _StagedBatch:
    value: Any
    coordinate_epoch: int
    delivered_batches: int
    sampler_checksum: int
    delivered_bitmap: bytes


@dataclass(frozen=True, slots=True)
class _Raised:
    error: BaseException


_END = object()


class PinnedDeliveryIterator(Iterator[Any]):
    """Stage exactly one speculative batch on a dedicated delivery thread."""

    def __init__(self, delivery: Any, iterator: Iterator[Any]) -> None:
        self._delivery = delivery
        self._iterator = iterator
        self._queue: Queue[object] = Queue(maxsize=1)
        self._acknowledged = Event()
        self._stopped = Event()
        self._thread: Thread | None = None
        self._complete = False
        self._coordinate_epoch = int(iterator.coordinate_epoch)
        self._delivered_batches = int(iterator.delivered_batches)
        self._sampler_checksum = int(iterator.sampler_checksum)
        self._delivered_bitmap = bytes(iterator.delivered_bitmap)
        self._delivery.bind_consumer_thread(get_ident())

    def __iter__(self) -> PinnedDeliveryIterator:
        return self

    def __next__(self) -> Any:
        if self._complete:
            raise StopIteration
        self._delivery.bind_consumer_thread(get_ident())
        self._start()
        item = self._queue.get()
        self._acknowledged.set()
        if item is _END:
            self._complete = True
            self._join_finished()
            raise StopIteration
        if isinstance(item, _Raised):
            self._complete = True
            self._join_finished()
            raise item.error
        if not isinstance(item, _StagedBatch):
            raise RuntimeError("staging thread returned an invalid delivery record")
        self._coordinate_epoch = item.coordinate_epoch
        self._delivered_batches = item.delivered_batches
        self._sampler_checksum = item.sampler_checksum
        self._delivered_bitmap = item.delivered_bitmap
        return item.value

    def _flush_telemetry(self) -> None:
        flush = getattr(self._iterator, "_flush_telemetry", None)
        if flush is not None:
            flush()

    @property
    def complete(self) -> bool:
        """Report the wrapped iterator's completion state."""
        return self._complete

    @property
    def coordinate_epoch(self) -> int:
        """Return the wrapped iterator's checkpoint epoch."""
        return self._coordinate_epoch

    @property
    def delivered_batches(self) -> int:
        """Return the wrapped iterator's delivered-batch prefix."""
        return self._delivered_batches

    @property
    def sampler_checksum(self) -> int:
        """Return the wrapped iterator's sampler-prefix checksum."""
        return self._sampler_checksum

    @property
    def delivered_bitmap(self) -> bytes:
        """Return the wrapped completion-order delivery bitmap."""
        return self._delivered_bitmap

    def capture_checkpoint(self) -> Any:
        """Return a wrapped iterable checkpoint without source callbacks."""
        return self._iterator.capture_checkpoint()

    def recover_lane(self, identity: int) -> None:
        """Forward one engine-signaled iterable lane recovery."""
        self._iterator.recover_lane(identity)

    def invalidate(self) -> None:
        """Invalidate the wrapped execution iterator."""
        self._stopped.set()
        self._acknowledged.set()
        self._iterator.invalidate()

    def close(self) -> None:
        """Stop and join the dedicated staging thread."""
        self.invalidate()
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join()
        self._thread = None

    def _start(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(
            target=self._produce,
            name="hyperloader-pinned-staging",
            daemon=True,
        )
        self._thread.start()

    def _produce(self) -> None:
        while not self._stopped.is_set():
            try:
                value = next(self._iterator)
                item: object = _StagedBatch(
                    value=self._delivery.stage(value),
                    coordinate_epoch=int(self._iterator.coordinate_epoch),
                    delivered_batches=int(self._iterator.delivered_batches),
                    sampler_checksum=int(self._iterator.sampler_checksum),
                    delivered_bitmap=bytes(self._iterator.delivered_bitmap),
                )
            except StopIteration:
                item = _END
            except BaseException as error:
                item = _Raised(error)
            if self._stopped.is_set():
                return
            self._acknowledged.clear()
            self._queue.put(item)
            while not self._stopped.is_set():
                if self._acknowledged.wait():
                    break
            if item is _END or isinstance(item, _Raised):
                return

    def _join_finished(self) -> None:
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join()

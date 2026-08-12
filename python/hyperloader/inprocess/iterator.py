"""Synchronous map-style execution in the calling process."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from hyperloader.rng import _user_code_context

from ..process.sizing import delivery_length
from ..state import resume_sample_position
from .rng import InProcessRngSession


class InProcessIterator(Iterator[Any]):
    """Evaluate one batch at a time while restoring trainer RNG state."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self._epoch = loader._epoch
        self._runtime = loader._sampler_runtime
        self._complete = False
        self._valid = True
        self._delivered = loader._resume_cursor_batches
        self._ordinal = (
            self._runtime.start_batch
            if loader.batch_sampler is not None
            else self._delivered
        )
        self._length = 0 if self._runtime is not None else delivery_length(loader)
        self._position = (
            self._runtime.start_samples
            if loader.sampler is not None
            else (
                0
                if loader.batch_sampler is not None
                else resume_sample_position(loader, self._length)
            )
        )

    def __iter__(self) -> InProcessIterator:
        return self

    def __next__(self) -> Any:
        if not self._valid:
            raise RuntimeError("in-process iterator is no longer active")
        batch = self._next_indices()
        if batch is None:
            self._finish_epoch()
            raise StopIteration
        indices, coordinates = batch
        session = InProcessRngSession()
        try:
            values = [
                self._evaluate(session, coordinate, index)
                for coordinate, index in zip(coordinates, indices, strict=True)
            ]
            if self._loader.collate_fn is not None:
                sample = session.install_collate(
                    self._loader.root_seed, self._epoch, self._delivered
                )
                with _user_code_context(sample):
                    value = self._loader.collate_fn(
                        values
                        if self._loader.batch_size is not None
                        or self._loader.batch_sampler is not None
                        else values[0]
                    )
            else:
                value = (
                    values[0]
                    if self._loader.batch_size is None
                    and self._loader.batch_sampler is None
                    else self._loader._collate_batch(values)
                )
        except BaseException:
            self._loader.close()
            raise
        finally:
            session.close()
        self._commit(len(indices))
        return value

    def _next_indices(self) -> tuple[tuple[Any, ...], tuple[int, ...]] | None:
        if self._loader.batch_sampler is not None:
            if not self._runtime.has_batch(self._ordinal):
                return None
            indices = self._runtime.batch(self._ordinal)
            if not indices:
                raise ValueError("user batch_sampler yielded an empty batch")
            start = self._runtime.sample_offset(self._ordinal)
            return indices, tuple(start + offset for offset in range(len(indices)))
        if self._loader.sampler is not None:
            width = self._loader.batch_size or 1
            indices = []
            while len(indices) < width and self._runtime.has_index(
                self._position + len(indices)
            ):
                indices.append(self._runtime.index(self._position + len(indices)))
            if not indices or (
                len(indices) < width
                and self._loader.batch_size is not None
                and self._loader.drop_last
            ):
                return None
            start = self._position
            return tuple(indices), tuple(
                start + offset for offset in range(len(indices))
            )
        if self._position >= self._length:
            return None
        width = self._loader.batch_size or 1
        stop = min(self._length, self._position + width)
        positions = tuple(range(self._position, stop))
        indices = tuple(
            self._loader._map_index(self._epoch, position) for position in positions
        )
        coordinates = tuple(
            self._loader._map_coordinate(position) for position in positions
        )
        return indices, coordinates

    def _evaluate(
        self, session: InProcessRngSession, coordinate: int, index: Any
    ) -> Any:
        sample = session.install(self._loader.root_seed, self._epoch, coordinate)
        with _user_code_context(sample):
            return self._loader._execution_dataset[index]

    def _commit(self, samples: int) -> None:
        if self._loader.batch_sampler is not None:
            self._ordinal += 1
        else:
            self._position += samples
        self._delivered += 1
        self._loader._epoch_state.mark_delivered(self._epoch)

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
        """Return the delivered batch prefix."""
        return self._delivered

    @property
    def sampler_checksum(self) -> int:
        """Return the checksum through the delivered sampler prefix."""
        if self._loader.batch_sampler is not None:
            return self._runtime.checksum_at(self._ordinal)
        if self._loader.sampler is not None:
            return self._runtime.checksum_at(self._position)
        return 0

    @property
    def delivered_bitmap(self) -> bytes:
        """Return the synchronous delivery bitmap sentinel."""
        return b""

    def invalidate(self) -> None:
        """Prevent a replaced iterator from delivering more values."""
        self._valid = False

    def _finish_epoch(self) -> None:
        if self._complete:
            return
        self._loader._epoch_state.complete(self._epoch)
        self._complete = True

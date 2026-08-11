"""Sampler-owned stream iteration and rolling replay checksums."""

from __future__ import annotations

import pickle
import zlib
from collections.abc import Iterable, Iterator
from typing import Any


class SamplerRuntime:
    """Resolve a sampler lazily by absolute stream position."""

    def __init__(
        self,
        sampler: Iterable[Any],
        *,
        batch_size: int | None,
        drop_last: bool,
        cursor_batches: int,
        expected_checksum: int,
    ) -> None:
        try:
            length: int | None = len(sampler)  # type: ignore[arg-type]
        except TypeError:
            length = None
        if length is not None and length < 0:
            raise ValueError("user sampler length must be nonnegative")
        resolved_batch = batch_size or 1
        self.length = length
        self._iterator: Iterator[Any] = iter(sampler)
        self._indices: list[Any] = []
        self._checksums = [0]
        self._exhausted = False
        requested_start = cursor_batches * resolved_batch
        self._generate_to(requested_start, allow_end=True)
        self.start_samples = min(requested_start, len(self._indices))
        if self._exhausted:
            full_batches, remainder = divmod(len(self._indices), resolved_batch)
            available_batches = full_batches + int(bool(remainder) and not drop_last)
            if cursor_batches > available_batches:
                self._raise_mismatch()
        self._verify(expected_checksum, self._checksums[self.start_samples])

    def index(self, position: int) -> Any:
        """Return one sampler index after materializing only the required prefix."""
        if position < 0:
            raise IndexError("sampler position is outside the declared stream")
        self._generate_to(position + 1, allow_end=True)
        if position >= len(self._indices):
            raise IndexError("sampler position is outside the produced stream")
        return self._indices[position]

    def has_index(self, position: int) -> bool:
        """Report whether an index exists at one stream position."""
        if position < 0:
            return False
        self._generate_to(position + 1, allow_end=True)
        return position < len(self._indices)

    def checksum_at(self, delivered_samples: int) -> int:
        """Return the rolling checksum at one delivered sample prefix."""
        if delivered_samples < 0:
            raise ValueError("delivered sampler prefix is outside the stream")
        self._generate_to(delivered_samples, allow_end=False)
        return self._checksums[delivered_samples]

    def probe(self) -> tuple[int, Any] | None:
        """Return the first execution coordinate after the restored prefix."""
        if not self.has_index(self.start_samples):
            return None
        return self.start_samples, self.index(self.start_samples)

    def _generate_to(self, count: int, *, allow_end: bool) -> None:
        while len(self._indices) < count:
            if self._exhausted:
                if allow_end:
                    return
                raise RuntimeError("user sampler ended before the delivered prefix")
            try:
                index = next(self._iterator)
            except StopIteration as error:
                self._exhausted = True
                if allow_end:
                    return
                raise RuntimeError(
                    "user sampler ended before its declared length"
                ) from error
            self._indices.append(index)
            self._checksums.append(_update_checksum(self._checksums[-1], index))

    @staticmethod
    def _verify(expected: int, actual: int) -> None:
        if expected != actual:
            SamplerRuntime._raise_mismatch()

    @staticmethod
    def _raise_mismatch() -> None:
        raise ValueError(
            "sampler checksum mismatch while restoring the delivered prefix; "
            "the sampler is nondeterministic or this rank's state belongs to "
            "a different rank"
        )


class BatchSamplerRuntime:
    """Resolve variable user batches and retain their consumed sample coordinates."""

    def __init__(
        self,
        sampler: Iterable[Any],
        *,
        cursor_batches: int,
        expected_checksum: int,
    ) -> None:
        try:
            self.length: int | None = len(sampler)  # type: ignore[arg-type]
        except TypeError:
            self.length = None
        if self.length is not None and self.length < 0:
            raise ValueError("user batch_sampler length must be nonnegative")
        self.start_batch = cursor_batches
        self._iterator: Iterator[Any] = iter(sampler)
        self._batches: list[tuple[Any, ...]] = []
        self._sample_offsets = [0]
        self._checksums = [0]
        self._exhausted = False
        self._generate_to(cursor_batches, allow_end=True)
        if len(self._batches) < cursor_batches:
            SamplerRuntime._raise_mismatch()
        SamplerRuntime._verify(expected_checksum, self._checksums[cursor_batches])

    def batch(self, ordinal: int) -> tuple[Any, ...]:
        """Return one exact user batch in sampler order."""
        if ordinal < 0:
            raise IndexError("batch sampler position is outside the declared stream")
        self._generate_to(ordinal + 1, allow_end=True)
        if ordinal >= len(self._batches):
            raise IndexError("batch sampler position is outside the produced stream")
        return self._batches[ordinal]

    def has_batch(self, ordinal: int) -> bool:
        """Report whether one user batch exists at an ordinal."""
        if ordinal < 0:
            return False
        self._generate_to(ordinal + 1, allow_end=True)
        return ordinal < len(self._batches)

    def sample_offset(self, ordinal: int) -> int:
        """Return the sample-stream coordinate at one batch boundary."""
        self._generate_to(ordinal, allow_end=False)
        return self._sample_offsets[ordinal]

    def checksum_at(self, delivered_batches: int) -> int:
        """Return the rolling checksum at one delivered batch prefix."""
        if delivered_batches < 0:
            raise ValueError("delivered batch-sampler prefix is outside the stream")
        self._generate_to(delivered_batches, allow_end=False)
        return self._checksums[delivered_batches]

    def probe(self) -> tuple[int, Any] | None:
        """Return the first sample coordinate and index after the restored prefix."""
        if not self.has_batch(self.start_batch):
            return None
        batch = self.batch(self.start_batch)
        if not batch:
            raise ValueError("user batch_sampler yielded an empty batch")
        return self.sample_offset(self.start_batch), batch[0]

    def _generate_to(self, count: int, *, allow_end: bool) -> None:
        while len(self._batches) < count:
            if self._exhausted:
                if allow_end:
                    return
                raise RuntimeError(
                    "user batch_sampler ended before the delivered prefix"
                )
            try:
                raw_batch = next(self._iterator)
            except StopIteration as error:
                self._exhausted = True
                if allow_end:
                    return
                raise RuntimeError(
                    "user batch_sampler ended before its declared length"
                ) from error
            try:
                batch = tuple(raw_batch)
            except TypeError as error:
                raise TypeError(
                    "user batch_sampler batches must be iterable"
                ) from error
            checksum = _update_boundary(self._checksums[-1], len(batch))
            for index in batch:
                checksum = _update_checksum(checksum, index)
            self._batches.append(batch)
            self._sample_offsets.append(self._sample_offsets[-1] + len(batch))
            self._checksums.append(checksum)


def build_sampler_runtime(loader: Any) -> SamplerRuntime | BatchSamplerRuntime:
    """Build the sampler cursor selected by one validated loader surface."""
    if loader.batch_sampler is not None:
        return BatchSamplerRuntime(
            loader.batch_sampler,
            cursor_batches=loader._resume_cursor_batches,
            expected_checksum=loader._resume_sampler_checksum,
        )
    return SamplerRuntime(
        loader.sampler,
        batch_size=loader.batch_size,
        drop_last=loader.drop_last,
        cursor_batches=loader._resume_cursor_batches,
        expected_checksum=loader._resume_sampler_checksum,
    )


def _update_checksum(checksum: int, value: Any) -> int:
    payload = pickle.dumps(value, protocol=5)
    checksum = zlib.crc32(len(payload).to_bytes(8, "little"), checksum)
    return zlib.crc32(payload, checksum)


def _update_boundary(checksum: int, length: int) -> int:
    return zlib.crc32(b"batch" + length.to_bytes(8, "little"), checksum)

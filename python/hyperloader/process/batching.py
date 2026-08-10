"""Homogeneous NumPy batch materialization inside one process worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hyperloader import _hyperloader

from .serialization import ResultEncoder

BATCH_MAGIC = b"HLBAT\x00"


@dataclass(frozen=True)
class EncodedCompletion:
    """One dispatch and its encoded successful or exceptional outcome."""

    dispatch: Any
    status: int
    payload: bytes


class ArrayBatcher:
    """Fuse one exact homogeneous NumPy batch or retain sample delivery."""

    def __init__(self, batch_size: int | None, encoder: ResultEncoder) -> None:
        self._enabled = batch_size is not None and batch_size > 1
        self._encoder = encoder
        self._entries: list[tuple[Any | None, Any]] = []
        self._disabled = False

    def seed_probe(self, value: Any) -> None:
        """Retain the already-executed first value without duplicating user code."""
        if not self._enabled:
            return
        if _eligible(value, None):
            self._entries.append((None, value))
        else:
            self._disabled = True

    def success(self, dispatch: Any, value: Any) -> list[EncodedCompletion]:
        """Accept one value and return every completion now publishable."""
        if not self._enabled:
            return [self._encoded(dispatch, value)]
        if self._disabled:
            result = [self._encoded(dispatch, value)]
            if dispatch.batch_end:
                self._reset()
            return result
        reference = None if not self._entries else self._entries[0][1]
        if not _eligible(value, reference):
            result = self._flush_samples()
            result.append(self._encoded(dispatch, value))
            self._disabled = not dispatch.batch_end
            return result
        self._entries.append((dispatch, value))
        if not dispatch.batch_end:
            return []
        values = [item for _, item in self._entries]
        commands = [command for command, _ in self._entries if command is not None]
        try:
            batch = _hyperloader._default_collate(values)
        except BaseException:
            return self._flush_samples()
        result = [EncodedCompletion(command, 0, b"") for command in commands[:-1]]
        result.append(
            EncodedCompletion(
                commands[-1],
                0,
                BATCH_MAGIC + self._encoder.encode_uncached(batch),
            )
        )
        self._reset()
        return result

    def failure(
        self, dispatch: Any, status: int, payload: bytes
    ) -> list[EncodedCompletion]:
        """Flush preceding values and retain the exact failing position."""
        result = self._flush_samples()
        result.append(EncodedCompletion(dispatch, status, payload))
        self._disabled = not dispatch.batch_end
        return result

    def _encoded(self, dispatch: Any, value: Any) -> EncodedCompletion:
        return EncodedCompletion(dispatch, 0, self._encoder.encode(value))

    def _flush_samples(self) -> list[EncodedCompletion]:
        result = [
            self._encoded(command, value)
            for command, value in self._entries
            if command is not None
        ]
        self._entries.clear()
        self._disabled = False
        return result

    def _reset(self) -> None:
        self._entries.clear()
        self._disabled = False


def _eligible(value: Any, reference: Any | None) -> bool:
    try:
        import numpy as np
    except ImportError:
        return False
    if (
        type(value) is not np.ndarray
        or not value.flags.c_contiguous
        or value.dtype.hasobject
    ):
        return False
    return reference is None or (
        value.dtype == reference.dtype and value.shape == reference.shape
    )


def unwrap_batch_payload(payload: bytes) -> bytes | None:
    """Return the encoded batch body or report an ordinary sample payload."""
    return payload[len(BATCH_MAGIC) :] if payload.startswith(BATCH_MAGIC) else None

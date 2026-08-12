"""Batch values and feeder adapters for live training cells."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenBatch:
    """One pre-tokenized batch plus its precomputed content digest."""

    tokens: torch.Tensor
    digest: str

    def validate(self) -> None:
        """Require a batch-shaped token tensor and a SHA-256 content digest."""
        if self.tokens.ndim != 2 or self.tokens.shape[0] <= 0:
            raise ValueError("token batches require a positive batch and sequence axis")
        if len(self.digest) != 64:
            raise ValueError("token batch digest must be a SHA-256 hexadecimal string")
        try:
            decoded = bytes.fromhex(self.digest)
        except ValueError as error:
            raise ValueError(
                "token batch digest must be a SHA-256 hexadecimal string"
            ) from error
        if len(decoded) != 32:
            raise ValueError("token batch digest must be a SHA-256 hexadecimal string")

    @property
    def samples(self) -> int:
        """Return the number of training samples in the batch."""
        return int(self.tokens.shape[0])

    def pin_memory(self) -> TokenBatch:
        """Return the same logical batch backed by accelerator-pinned host memory."""
        return TokenBatch(self.tokens.pin_memory(), self.digest)


def collate_token_batch(rows: list[torch.Tensor]) -> TokenBatch:
    """Stack pre-tokenized samples and hash the delivered tensor exactly."""
    if not rows:
        raise ValueError("token collation requires at least one sample")
    tokens = torch.stack(rows)
    digest = hashlib.sha256()
    digest.update(str(tokens.dtype).encode())
    digest.update(str(tuple(tokens.shape)).encode())
    digest.update(tokens.detach().cpu().contiguous().numpy().tobytes())
    batch = TokenBatch(tokens, digest.hexdigest())
    batch.validate()
    return batch


class ResidentTokenFeeder:
    """Cycle through a pre-materialized batch bank without loader work."""

    def __init__(self, system: str, batches: tuple[TokenBatch, ...]) -> None:
        if not system or not batches:
            raise ValueError("resident feeders require a system name and batch bank")
        for batch in batches:
            batch.validate()
        self.system = system
        self._batches = batches
        self._index = 0

    def next_batch(self) -> TokenBatch:
        """Return the next resident batch without rebuilding an iterator."""
        batch = self._batches[self._index]
        self._index = (self._index + 1) % len(self._batches)
        return batch


class IteratorTokenFeeder:
    """Adapt one already-open loader iterator without hiding exhaustion."""

    def __init__(self, system: str, iterator: Iterator[TokenBatch]) -> None:
        if not system:
            raise ValueError("iterator feeders require a system name")
        self.system = system
        self._iterator = iterator

    def next_batch(self) -> TokenBatch:
        """Return and validate one batch from the installed loader path."""
        batch = next(self._iterator)
        batch.validate()
        return batch

"""Deterministic pre-tokenized rows for live training measurements."""

from __future__ import annotations

import hashlib

import torch
from torch.utils.data import Dataset


class PretokenizedRows(Dataset[torch.Tensor]):
    """Hold one finite, repeatable bank of already-tokenized samples."""

    def __init__(
        self,
        *,
        rows: int,
        sequence_length: int,
        vocabulary_size: int,
        seed: int,
    ) -> None:
        if rows <= 0 or sequence_length < 2 or vocabulary_size <= 1:
            raise ValueError("token source dimensions must be positive")
        generator = torch.Generator().manual_seed(seed)
        self._tokens = torch.randint(
            0,
            vocabulary_size,
            (rows, sequence_length),
            generator=generator,
            dtype=torch.int64,
        )

    def __len__(self) -> int:
        return int(self._tokens.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._tokens[index]

    @property
    def tensor(self) -> torch.Tensor:
        """Return the retained contiguous token storage for native batch views."""
        return self._tokens

    @property
    def identity(self) -> str:
        """Return a digest of the exact finite token source."""
        digest = hashlib.sha256()
        digest.update(str(self._tokens.dtype).encode())
        digest.update(str(tuple(self._tokens.shape)).encode())
        digest.update(self._tokens.numpy().tobytes())
        return digest.hexdigest()


def token_source_identity(
    *, rows: int, sequence_length: int, vocabulary_size: int, seed: int
) -> str:
    """Materialize the specified token source and return its exact digest."""
    return PretokenizedRows(
        rows=rows,
        sequence_length=sequence_length,
        vocabulary_size=vocabulary_size,
        seed=seed,
    ).identity

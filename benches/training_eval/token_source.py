"""Deterministic pre-tokenized rows for live training measurements."""

from __future__ import annotations

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

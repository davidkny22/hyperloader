"""Ordinary map pipelines for fallback no-regression measurement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WorkloadSpec:
    """One deterministic dataset family and its batch checksum rule."""

    name: str

    def dataset(self, length: int) -> object:
        """Construct the named finite map dataset."""
        if self.name == "fixed-record":
            return FixedRecordDataset(length)
        if self.name == "numpy-array":
            return NumpyArrayDataset(length)
        raise ValueError(f"unknown fallback workload {self.name}")

    def checksum(self, batch: Any) -> int:
        """Read one delivered scalar so both systems prove matching work."""
        if self.name == "fixed-record":
            return int(batch["tokens"][0][0]) + int(batch["label"][0])
        return int(batch[0, 0])


class FixedRecordDataset:
    """Return a small token record built by ordinary Python code."""

    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "tokens": [((index * 131) + offset * 17) % 32_003 for offset in range(32)],
            "label": index % 17,
        }


class NumpyArrayDataset:
    """Return a fixed-shape numeric array from deterministic NumPy work."""

    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> np.ndarray:
        values = np.arange(256, dtype=np.float32)
        values += np.float32(index % 251)
        return values


WORKLOADS = (WorkloadSpec("fixed-record"), WorkloadSpec("numpy-array"))

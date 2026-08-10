"""Transport-bound identity-configuration benchmark datasets."""

from __future__ import annotations

from typing import Any

import numpy as np

from overhead_feeders import BATCH_SIZE, SAMPLE_WIDTH


class FixedTextDataset:
    """Expose pre-tokenized tensor rows through an unrecognized dataset type."""

    def __init__(self, rows: int) -> None:
        import torch

        source = torch.arange(SAMPLE_WIDTH, dtype=torch.int64)
        self._values = source.repeat(rows, 1)

    def __len__(self) -> int:
        return self._values.shape[0]

    def __getitem__(self, index: int) -> Any:
        return self._values[index]


class NumpyArrayDataset:
    """Expose dense NumPy rows through the black-box process contract."""

    def __init__(self, rows: int) -> None:
        source = np.arange(SAMPLE_WIDTH, dtype=np.int64)
        self._values = np.repeat(source[None, :], rows, axis=0)

    def __len__(self) -> int:
        return self._values.shape[0]

    def __getitem__(self, index: int) -> np.ndarray[Any, np.dtype[np.int64]]:
        return self._values[index]


class ArrowTabularDataset:
    """Expose fixed-width Arrow rows as dense arrays for default collation."""

    def __init__(self, rows: int) -> None:
        import pyarrow as pa

        source = np.arange(SAMPLE_WIDTH, dtype=np.int64)
        values = np.repeat(source[None, :], rows, axis=0)
        self._values = pa.FixedSizeListArray.from_arrays(
            pa.array(values.reshape(-1)), SAMPLE_WIDTH
        )

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> np.ndarray[Any, np.dtype[np.int64]]:
        values = self._values[index].values.to_numpy(zero_copy_only=True)
        return np.array(values, dtype=np.int64, copy=True)


def make_identity_dataset(workload: str, batch_count: int) -> Any:
    """Build one 8x-LLC-sized dataset for a named transport cell."""
    if batch_count <= 0:
        raise ValueError("resident batch count must be positive")
    rows = batch_count * BATCH_SIZE
    factories = {
        "fixed-text": FixedTextDataset,
        "numpy-array": NumpyArrayDataset,
        "arrow-tabular": ArrowTabularDataset,
    }
    try:
        factory = factories[workload]
    except KeyError as error:
        raise ValueError(f"unknown identity workload {workload!r}") from error
    return factory(rows)

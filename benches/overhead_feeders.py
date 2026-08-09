"""Resident and public-loader feeders for the Spark overhead cells."""

from __future__ import annotations

import itertools
import math
from typing import Any

BATCH_SIZE = 64
SAMPLE_WIDTH = 512
WORKERS = 4
FRONTIER_DEPTH = 128
LLC_MULTIPLIER = 8


def fixed_text_tensor(batch_count: int) -> Any:
    """Materialize one contiguous pre-tokenized tensor working set."""
    import torch

    if batch_count <= 0:
        raise ValueError("resident batch count must be positive")
    source = torch.arange(SAMPLE_WIDTH, dtype=torch.int64)
    return source.repeat(BATCH_SIZE * batch_count, 1)


class LoaderFeeder:
    """Cycle batches through the installed public DataLoader path."""

    def __init__(self, dataset: Any) -> None:
        from hyperloader import DataLoader
        from hyperloader.config import HyperConfig, SchedulerConfig

        self._loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=WORKERS,
            config=HyperConfig(
                scheduler=SchedulerConfig(frontier_depth=FRONTIER_DEPTH)
            ),
        )
        self._iterator = iter(self._loader)
        self._batch_count = len(dataset) // BATCH_SIZE
        self.batches = 0

    def next_batch(self) -> Any:
        """Return the next public-path batch, cycling complete epochs."""
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._loader)
            batch = next(self._iterator)
        self.batches += 1
        return batch

    def warm(self) -> None:
        """Touch a full resident working set outside measurement."""
        for _ in range(self._batch_count):
            self.next_batch()

    def close(self) -> None:
        """Release persistent workers and arena ownership."""
        self._loader.close()


class ResidentFeeder:
    """Cycle the same already-collated resident batches without loader work."""

    def __init__(self, dataset: Any) -> None:
        self._batches = [
            dataset[start : start + BATCH_SIZE]
            for start in range(0, len(dataset), BATCH_SIZE)
        ]
        self._iterator = itertools.cycle(self._batches)
        self.batches = 0

    def next_batch(self) -> Any:
        """Return one resident collated batch."""
        self.batches += 1
        return next(self._iterator)

    def warm(self) -> None:
        """Touch every resident batch outside measurement."""
        for _ in range(len(self._batches)):
            self.next_batch()


def tensor_sizes(dataset: Any) -> tuple[int, int]:
    """Measure logical sample and batch bytes for the tensor-view plan."""
    sample = dataset[0]
    logical = sample.numel() * sample.element_size()
    batch = logical * BATCH_SIZE
    return logical, batch


def resident_batch_count(llc_bytes: int, batch_bytes: int) -> int:
    """Size the resident ring to at least eight times total LLC."""
    if llc_bytes <= 0 or batch_bytes <= 0:
        raise ValueError("LLC and batch byte counts must be positive")
    return math.ceil(LLC_MULTIPLIER * llc_bytes / batch_bytes)

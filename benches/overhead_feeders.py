"""Resident and public-loader feeders for the Spark overhead cells."""

from __future__ import annotations

import itertools
import math
import os
import pickle
from typing import Any

BATCH_SIZE = 64
SAMPLE_WIDTH = 512
WORKERS = 4
FRONTIER_DEPTH = 128
LLC_MULTIPLIER = 8


class FixedTextDataset:
    """Pre-materialized independent token tensors for black-box execution."""

    def __init__(self, batch_count: int) -> None:
        import torch

        if batch_count <= 0:
            raise ValueError("resident batch count must be positive")
        source = torch.arange(SAMPLE_WIDTH, dtype=torch.int64)
        self._samples = [source.clone() for _ in range(BATCH_SIZE * batch_count)]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Any:
        return self._samples[index]


def pin_efficiency_worker(worker: int) -> None:
    """Keep each process worker on one Spark efficiency core."""
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {worker})


class LoaderFeeder:
    """Cycle batches through the installed public DataLoader path."""

    def __init__(self, dataset: FixedTextDataset) -> None:
        from hyperloader import DataLoader
        from hyperloader.config import HyperConfig, SchedulerConfig

        self._loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=WORKERS,
            worker_init_fn=pin_efficiency_worker,
            multiprocessing_context="forkserver",
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

    def __init__(self, dataset: FixedTextDataset) -> None:
        import torch

        samples = [dataset[index] for index in range(len(dataset))]
        self._batches = [
            torch.stack(samples[start : start + BATCH_SIZE])
            for start in range(0, len(samples), BATCH_SIZE)
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


def payload_sizes(dataset: FixedTextDataset) -> tuple[int, int, int]:
    """Measure logical sample, serialized sample, and collated batch bytes."""
    sample = dataset[0]
    logical = sample.numel() * sample.element_size()
    serialized = len(pickle.dumps(sample, protocol=5))
    batch = logical * BATCH_SIZE
    return logical, serialized, batch


def resident_batch_count(llc_bytes: int, batch_bytes: int) -> int:
    """Size the resident ring to at least eight times total LLC."""
    if llc_bytes <= 0 or batch_bytes <= 0:
        raise ValueError("LLC and batch byte counts must be positive")
    return math.ceil(LLC_MULTIPLIER * llc_bytes / batch_bytes)

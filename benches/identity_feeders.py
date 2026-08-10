"""Equal-tuning torch and hyperloader process feeders."""

from __future__ import annotations

import os
import time
from typing import Any

from overhead_feeders import BATCH_SIZE, WORKERS

PREFETCH_FACTOR = 2
EQUAL_FRONTIER_DEPTH = PREFETCH_FACTOR * WORKERS * BATCH_SIZE


def pin_efficiency_worker(worker: int) -> None:
    """Pin corresponding workers from both systems to the same efficiency cores."""
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {worker})


def torch_loader_arguments() -> dict[str, Any]:
    """Return the fixed equal-tuning torch process controls."""
    return {
        "batch_size": BATCH_SIZE,
        "num_workers": WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
        "persistent_workers": True,
        "worker_init_fn": pin_efficiency_worker,
        "multiprocessing_context": "forkserver",
    }


def hyperloader_arguments() -> dict[str, Any]:
    """Return the matching hyperloader identity controls."""
    from hyperloader.config import HyperConfig, SchedulerConfig

    return {
        "batch_size": BATCH_SIZE,
        "num_workers": WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
        "persistent_workers": True,
        "worker_init_fn": pin_efficiency_worker,
        "multiprocessing_context": "forkserver",
        "config": HyperConfig(
            scheduler=SchedulerConfig(frontier_depth=EQUAL_FRONTIER_DEPTH)
        ),
    }


class TorchFeeder:
    """Cycle batches through torch's persistent process DataLoader."""

    def __init__(self, dataset: Any) -> None:
        import torch

        started = time.perf_counter()
        self._loader = torch.utils.data.DataLoader(dataset, **torch_loader_arguments())
        self._iterator = iter(self._loader)
        self.startup_seconds = time.perf_counter() - started
        self._batch_count = len(dataset) // BATCH_SIZE
        self.batches = 0

    def next_batch(self) -> Any:
        """Return the next batch and cycle complete epochs."""
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._loader)
            batch = next(self._iterator)
        self.batches += 1
        return batch

    def warm(self) -> None:
        """Touch one full dataset outside measurement."""
        for _ in range(self._batch_count):
            self.next_batch()

    def close(self) -> None:
        """Release torch's persistent process workers."""
        shutdown = getattr(self._iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()


class HyperloaderFeeder:
    """Cycle batches through hyperloader's public identity configuration."""

    def __init__(self, dataset: Any) -> None:
        from hyperloader import DataLoader
        from hyperloader.planner import BlackBoxPlan

        started = time.perf_counter()
        self._loader = DataLoader(dataset, **hyperloader_arguments())
        if not isinstance(self._loader._plan, BlackBoxPlan):
            self._loader.close()
            raise RuntimeError("identity workload did not select the black-box plan")
        self._iterator = iter(self._loader)
        self.startup_seconds = time.perf_counter() - started
        self._batch_count = len(dataset) // BATCH_SIZE
        self.batches = 0

    def next_batch(self) -> Any:
        """Return the next batch and cycle complete epochs."""
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._loader)
            batch = next(self._iterator)
        self.batches += 1
        return batch

    def warm(self) -> None:
        """Touch one full dataset outside measurement."""
        for _ in range(self._batch_count):
            self.next_batch()

    def close(self) -> None:
        """Release hyperloader's persistent process workers."""
        self._loader.close()

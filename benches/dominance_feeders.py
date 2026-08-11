"""Equal-input torch, SPDL, and hyperloader benchmark feeders."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Literal

from dominance_protocol import SelectedConfig
from dominance_workloads import WorkloadBundle

EFFICIENCY_CORES = tuple(range(10))


def pin_efficiency_worker(worker: int) -> None:
    """Pin process workers to the measured efficiency-core cluster."""
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {EFFICIENCY_CORES[worker % len(EFFICIENCY_CORES)]})


@contextmanager
def native_thread_affinity() -> Any:
    """Make newly created native threads inherit the efficiency-core mask."""
    if not hasattr(os, "sched_getaffinity"):
        yield
        return
    original = os.sched_getaffinity(0)
    available = set(EFFICIENCY_CORES).intersection(original)
    if not available:
        available = set(original)
    os.sched_setaffinity(0, available)
    try:
        yield
    finally:
        os.sched_setaffinity(0, original)


class TorchFeeder:
    """Cycle one stock torch process loader under a selected configuration."""

    system = "torch"

    def __init__(self, workload: WorkloadBundle, selected: SelectedConfig) -> None:
        import torch

        started = time.perf_counter()
        self._workload = workload
        self._loader = torch.utils.data.DataLoader(
            workload.reference_dataset,
            batch_size=workload.batch_size,
            num_workers=selected.workers,
            prefetch_factor=selected.prefetch_factor,
            persistent_workers=True,
            worker_init_fn=pin_efficiency_worker,
            multiprocessing_context="forkserver",
            collate_fn=workload.collate_fn,
            in_order=True,
        )
        self._iterator = iter(self._loader)
        self.startup_seconds = time.perf_counter() - started
        self.batches = 0

    def next_batch(self) -> Any:
        """Return one dense batch and cycle complete epochs."""
        try:
            value = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._loader)
            value = next(self._iterator)
        self.batches += 1
        return self._workload.normalize(value)

    def close(self) -> None:
        """Stop persistent reference workers."""
        shutdown = getattr(self._iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()

    def report(self) -> dict[str, object]:
        """Describe the reference delivery state used by this feeder."""
        return {"batches": self.batches, "pin_memory": False}


class HyperloaderFeeder:
    """Cycle the installed public loader under a selected configuration."""

    system = "hyperloader"

    def __init__(
        self,
        workload: WorkloadBundle,
        selected: SelectedConfig,
        *,
        delivery_memory: Literal["auto", "host", "pinned"] = "auto",
    ) -> None:
        from hyperloader import DataLoader
        from hyperloader.config import (
            DeterminismConfig,
            HyperConfig,
            MemoryConfig,
            SchedulerConfig,
        )

        started = time.perf_counter()
        frontier = selected.workers * selected.prefetch_factor * workload.batch_size
        determinism = (
            DeterminismConfig(decoder_pins=workload.decoder_pins)
            if workload.decoder_pins is not None
            else DeterminismConfig()
        )
        with native_thread_affinity():
            self._loader = DataLoader(
                workload.hyperloader_dataset,
                batch_size=workload.batch_size,
                num_workers=selected.workers,
                prefetch_factor=selected.prefetch_factor,
                persistent_workers=True,
                worker_init_fn=pin_efficiency_worker,
                multiprocessing_context="forkserver",
                config=HyperConfig(
                    determinism=determinism,
                    memory=MemoryConfig(delivery_memory=delivery_memory),
                    scheduler=SchedulerConfig(
                        frontier_depth=frontier,
                        profile_cache="off",
                    ),
                ),
            )
            self._iterator = iter(self._loader)
        self._workload = workload
        self.startup_seconds = time.perf_counter() - started
        self.batches = 0

    def next_batch(self) -> Any:
        """Return one dense public-path batch and cycle complete epochs."""
        try:
            value = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._loader)
            value = next(self._iterator)
        self.batches += 1
        return self._workload.normalize(value)

    def close(self) -> None:
        """Release native workers and named arena ownership."""
        self._loader.close()

    def report(self) -> dict[str, object]:
        """Capture public loader instruments before ownership closes."""
        return {
            "batches": self.batches,
            "delivery_memory": self._loader.delivery_memory,
            "stats": self._loader.stats(),
        }


class SpdlFeeder:
    """Cycle one explicit SPDL thread pipeline under a selected configuration."""

    system = "spdl"

    def __init__(self, workload: WorkloadBundle, selected: SelectedConfig) -> None:
        from spdl.dataloader import DataLoader

        started = time.perf_counter()
        self._workload = workload
        self._selected = selected
        self._loader = DataLoader(
            range(len(workload.reference_dataset)),
            preprocessor=workload.reference_dataset.__getitem__,
            batch_size=workload.batch_size,
            aggregator=workload.collate_fn,
            buffer_size=selected.workers * selected.prefetch_factor,
            num_threads=selected.workers,
            output_order="input",
        )
        self._iterator: Any = None
        self._pending: Any = None
        self._start_iterator()
        self.startup_seconds = time.perf_counter() - started
        self.batches = 0

    def _start_iterator(self) -> None:
        with native_thread_affinity():
            self._iterator = iter(self._loader)
            self._pending = next(self._iterator)

    def next_batch(self) -> Any:
        """Return one dense thread-pipeline batch and cycle complete epochs."""
        if self._pending is not None:
            value = self._pending
            self._pending = None
        else:
            try:
                value = next(self._iterator)
            except StopIteration:
                self._start_iterator()
                value = self._pending
                self._pending = None
        self.batches += 1
        return self._workload.normalize(value)

    def close(self) -> None:
        """Stop the active SPDL pipeline generator."""
        close = getattr(self._iterator, "close", None)
        if close is not None:
            close()

    def report(self) -> dict[str, object]:
        """Describe the completed thread-pipeline delivery count."""
        return {"batches": self.batches}


def build_feeder(
    system: str,
    workload: WorkloadBundle,
    selected: SelectedConfig,
    *,
    delivery_memory: Literal["auto", "host", "pinned"] = "auto",
) -> TorchFeeder | HyperloaderFeeder | SpdlFeeder:
    """Construct one named system through its public entry point."""
    if system == "hyperloader":
        return HyperloaderFeeder(
            workload,
            selected,
            delivery_memory=delivery_memory,
        )
    if delivery_memory != "auto":
        raise ValueError("delivery memory overrides apply only to hyperloader")
    factories = {
        "torch": TorchFeeder,
        "spdl": SpdlFeeder,
    }
    try:
        factory = factories[system]
    except KeyError as error:
        raise ValueError(f"unknown dominance system {system!r}") from error
    return factory(workload, selected)

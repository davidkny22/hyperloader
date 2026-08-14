"""Public loader adapters for live training comparisons."""

from __future__ import annotations

from collections.abc import Callable, Sized
from pathlib import Path
from typing import Any, Protocol

from .controls.processes import (
    process_record,
    validate_worker_probes,
    worker_probe_records,
)
from .controls.worker_probe import WorkerEnvironmentProbe


class TrainingBatch(Protocol):
    """Validated batch metadata consumed by the live training loop."""

    digest: str

    @property
    def samples(self) -> int:
        """Return the number of samples delivered in the batch."""

    def validate(self) -> None:
        """Reject a malformed batch before the timed consumer sees it."""


class PublicLoaderFeeder:
    """Cycle one already-constructed public loader over complete epochs."""

    def __init__(
        self,
        system: str,
        loader: Any,
        worker_count: int,
        prefetch: int,
        worker_environment_dir: Path | None,
        batch_adapter: Callable[[Any], TrainingBatch] | None = None,
    ) -> None:
        self.system = system
        self.worker_count = worker_count
        self.prefetch = prefetch
        self._worker_environment_dir = worker_environment_dir
        self._batch_adapter = batch_adapter
        self._loader = loader
        self._iterator = iter(loader)

    def next_batch(self) -> TrainingBatch:
        """Return one validated batch and reopen only at an epoch boundary."""
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._loader)
            batch = next(self._iterator)
        if self._batch_adapter is not None:
            batch = self._batch_adapter(batch)
        batch.validate()
        return batch

    def close(self) -> None:
        """Release public loader resources without assuming one implementation."""
        close = getattr(self._loader, "close", None)
        if close is not None:
            close()
            return
        shutdown = getattr(self._iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
        iterator_close = getattr(self._iterator, "close", None)
        if iterator_close is not None:
            iterator_close()

    def state_dict(self) -> dict[str, object]:
        """Capture an exact public loader coordinate when the system supports it."""
        capture = getattr(self._loader, "state_dict", None)
        if capture is None:
            raise RuntimeError(f"{self.system} does not expose resumable loader state")
        return capture()

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore a public loader coordinate and reopen its iterator."""
        restore = getattr(self._loader, "load_state_dict", None)
        if restore is None:
            raise RuntimeError(f"{self.system} does not expose resumable loader state")
        restore(state)
        self._iterator = iter(self._loader)

    def control_snapshot(self) -> dict[str, Any]:
        """Return live process and worker-boot evidence for this feeder."""
        pids = self._worker_pids()
        expected_workers = len(pids)
        records = worker_probe_records(
            self._worker_environment_dir, expected_workers=expected_workers
        )
        if self.system in {"torch", "hyperloader"}:
            validate_worker_probes(records, expected_workers=expected_workers)
        return {
            "configured_prefetch": self.prefetch,
            "configured_workers": self.worker_count,
            "processes": [process_record(pid) for pid in pids],
            "system": self.system,
            "worker_boot": records,
        }

    def _worker_pids(self) -> tuple[int, ...]:
        if self.system == "hyperloader":
            pool = getattr(self._loader, "_process_pool", None)
            if pool is not None:
                return tuple(pool.worker_pids)
        workers = getattr(self._iterator, "_workers", ())
        pids = tuple(
            int(worker.pid)
            for worker in workers
            if getattr(worker, "pid", None) is not None
        )
        return pids


def build_public_feeder(
    system: str,
    dataset: Sized,
    *,
    batch_size: int,
    workers: int,
    prefetch: int,
    collate: Callable[[list[Any]], TrainingBatch] | None,
    pin_memory: bool = False,
    worker_environment_dir: Path | None = None,
    batch_adapter: Callable[[Any], TrainingBatch] | None = None,
) -> PublicLoaderFeeder:
    """Construct Torch, hyperloader, or SPDL through its public import path."""
    _validate_controls(batch_size, workers, prefetch)
    if system == "torch":
        import torch

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            collate_fn=collate,
            pin_memory=pin_memory,
            worker_init_fn=_worker_probe(system, workers, worker_environment_dir),
            **_process_controls(workers, prefetch),
        )
    elif system == "hyperloader":
        from hyperloader import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            collate_fn=collate,
            pin_memory=pin_memory,
            worker_init_fn=_worker_probe(system, workers, worker_environment_dir),
            **_process_controls(workers, prefetch),
        )
    elif system == "spdl":
        if workers == 0:
            raise ValueError("SPDL reference execution requires at least one worker")
        if collate is None:
            raise ValueError("SPDL reference execution requires an aggregator")
        from spdl.dataloader import DataLoader

        aggregator = _pinning_collate(collate) if pin_memory else collate
        loader = DataLoader(
            range(len(dataset)),
            preprocessor=dataset.__getitem__,
            batch_size=batch_size,
            aggregator=aggregator,
            buffer_size=workers * prefetch,
            num_threads=workers,
            output_order="input",
        )
    else:
        raise ValueError(f"unknown training feeder {system!r}")
    evidence_dir = (
        worker_environment_dir / system if worker_environment_dir is not None else None
    )
    return PublicLoaderFeeder(
        system,
        loader,
        workers,
        prefetch,
        evidence_dir,
        batch_adapter,
    )


def _worker_probe(
    system: str, workers: int, directory: Path | None
) -> WorkerEnvironmentProbe | None:
    if workers == 0 or directory is None:
        return None
    return WorkerEnvironmentProbe(str(directory / system))


def _pinning_collate(
    collate: Callable[[list[Any]], TrainingBatch],
) -> Callable[[list[Any]], TrainingBatch]:
    def pinned(rows: list[Any]) -> TrainingBatch:
        batch = collate(rows)
        pin = getattr(batch, "pin_memory", None)
        if pin is None:
            raise TypeError("pinned SPDL delivery requires a pinnable training batch")
        return pin()

    return pinned


def _validate_controls(batch_size: int, workers: int, prefetch: int) -> None:
    if batch_size <= 0 or workers < 0 or prefetch <= 0:
        raise ValueError(
            "batch size and prefetch must be positive and workers nonnegative"
        )


def _process_controls(workers: int, prefetch: int) -> dict[str, Any]:
    if workers == 0:
        return {}
    return {"prefetch_factor": prefetch, "persistent_workers": True}

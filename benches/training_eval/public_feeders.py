"""Public loader adapters for live training comparisons."""

from __future__ import annotations

from collections.abc import Callable, Sized
from typing import Any, Protocol


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

    def __init__(self, system: str, loader: Any, worker_count: int) -> None:
        self.system = system
        self.worker_count = worker_count
        self._loader = loader
        self._iterator = iter(loader)

    def next_batch(self) -> TrainingBatch:
        """Return one validated batch and reopen only at an epoch boundary."""
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._loader)
            batch = next(self._iterator)
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


def build_public_feeder(
    system: str,
    dataset: Sized,
    *,
    batch_size: int,
    workers: int,
    prefetch: int,
    collate: Callable[[list[Any]], TrainingBatch],
    pin_memory: bool = False,
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
            **_process_controls(workers, prefetch),
        )
    elif system == "spdl":
        if workers == 0:
            raise ValueError("SPDL reference execution requires at least one worker")
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
    return PublicLoaderFeeder(system, loader, workers)


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

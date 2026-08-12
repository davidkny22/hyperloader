"""Torch-compatible execution in the calling process."""

from __future__ import annotations

import inspect
import weakref
from collections.abc import Iterator
from typing import Any

import torch

from hyperloader.config import AUTO
from hyperloader.epoch import EpochState
from hyperloader.fingerprint import (
    build_contract_fingerprint,
    build_dataset_fingerprint,
    require_fingerprint_match,
)

from .rng import (
    capture_generator,
    capture_globals,
    restore_generator,
    restore_globals,
)
from .state import CompatZeroCheckpoint


class CompatZeroIterator(Iterator[Any]):
    """Track delivery around torch's exact single-process iterator."""

    def __init__(
        self,
        loader: Any,
        iterator: Iterator[Any],
        iterator_globals: dict[str, bytes],
        iterator_generator: bytes | None,
        delivered: int = 0,
    ) -> None:
        self._loader = loader
        self._iterator = iterator
        self._iterator_globals = iterator_globals
        self._iterator_generator = iterator_generator
        self._delivered = delivered
        self._complete = False
        self._valid = True

    def __iter__(self) -> CompatZeroIterator:
        return self

    def __next__(self) -> Any:
        if not self._valid:
            raise RuntimeError("compat iterator is no longer active")
        try:
            value = next(self._iterator)
        except StopIteration:
            self._complete = True
            raise
        self._delivered += 1
        return value

    @property
    def complete(self) -> bool:
        """Report whether the torch iterator is exhausted."""
        return self._complete

    def capture_checkpoint(self) -> CompatZeroCheckpoint:
        """Capture delivered progress and ambient CPU RNG state."""
        return CompatZeroCheckpoint(
            delivered_batches=self._delivered,
            iterator_globals=self._iterator_globals,
            current_globals=capture_globals(),
            iterator_generator=self._iterator_generator,
            current_generator=capture_generator(self._loader._compat_generator),
            fingerprint=self._loader._fingerprint,
        )

    def invalidate(self) -> None:
        """Prevent further delivery from a replaced compat iterator."""
        self._valid = False

    def _flush_telemetry(self) -> None:
        """Keep the public telemetry snapshot hook available in compat mode."""


def prepare(loader: Any) -> None:
    """Build torch's zero-worker loader without native planning side effects."""
    workers = 0 if loader.num_workers is AUTO else int(loader.num_workers)
    if workers != 0:
        raise ValueError("zero-worker compatibility requires num_workers=0")
    loader.num_workers = 0
    if not loader._persistent_workers_explicit:
        loader.persistent_workers = False
    loader.prefetch_factor = None
    generator = loader.generator
    if generator is None and loader.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(loader.seed)
    loader._compat_generator = generator
    loader.root_seed = 0 if generator is None else int(generator.initial_seed())
    loader._epoch_state = EpochState()
    loader._resume_compat_state: CompatZeroCheckpoint | None = None
    loader._plan = (
        None
        if isinstance(loader.dataset, torch.utils.data.IterableDataset)
        else object()
    )
    loader._distributed_topology = None
    loader._map_placement = None
    loader._iterable_payload = None
    loader._memory_ledger = None
    loader._execution_dataset = loader.dataset
    loader._machine_identity = None
    loader._calibration = None
    loader._pinned_delivery = None
    loader._machine_keeper = None
    loader._machine_keeper_cpus = ()
    loader._machine_keeper_interrupt_cpus = ()
    loader._machine_keeper_consumer_cpu = None
    loader._machine_keeper_route_refresh_ns = 0
    loader._machine_keeper_route_batches = 0
    loader._machine_keeping_last_delivery_ns = 0
    loader._controller = None
    loader._last_frontier_report = None
    loader._last_controller_report = None
    loader._decoder_selections = ()
    loader._native_batch_probe = None
    loader._native_batch_shape = None
    loader._cost_profile = None
    loader._dataset_fingerprint = build_dataset_fingerprint(
        loader.dataset, loader.config.determinism.fingerprint
    )
    loader._fingerprint = build_contract_fingerprint(loader)
    loader._compat_loader = _torch_loader(loader)


def iterate(loader: Any) -> CompatZeroIterator:
    """Create one torch iterator, replaying a loaded delivered prefix if needed."""
    active = (
        None if loader._active_iterator_ref is None else loader._active_iterator_ref()
    )
    if active is not None and not active.complete:
        loader.close()
    resume = loader._resume_compat_state
    if resume is None:
        iterator_globals = capture_globals()
        iterator_generator = capture_generator(loader._compat_generator)
        delivered = 0
        torch_iterator = _start_torch_iterator(loader)
    else:
        restore_globals(resume.iterator_globals)
        restore_generator(loader._compat_generator, resume.iterator_generator)
        torch_iterator = _start_torch_iterator(loader)
        for _ in range(resume.delivered_batches):
            try:
                next(torch_iterator)
            except StopIteration as error:
                raise RuntimeError(
                    "compat source ended before its delivered cursor"
                ) from error
        restore_generator(loader._compat_generator, resume.current_generator)
        restore_globals(resume.current_globals)
        iterator_globals = resume.iterator_globals
        iterator_generator = resume.iterator_generator
        delivered = resume.delivered_batches
    wrapper = CompatZeroIterator(
        loader,
        torch_iterator,
        iterator_globals,
        iterator_generator,
        delivered,
    )
    loader._resume_compat_state = None
    loader._active_iterator_ref = weakref.ref(wrapper)
    return wrapper


def capture_state(loader: Any) -> dict[str, object]:
    """Capture active or pristine zero-worker compat state."""
    active = (
        None if loader._active_iterator_ref is None else loader._active_iterator_ref()
    )
    if active is not None and not active.complete:
        return active.capture_checkpoint().to_dict()
    if loader._resume_compat_state is not None:
        return loader._resume_compat_state.to_dict()
    globals_state = capture_globals()
    generator_state = capture_generator(loader._compat_generator)
    return CompatZeroCheckpoint(
        delivered_batches=0,
        iterator_globals=globals_state,
        current_globals=globals_state,
        iterator_generator=generator_state,
        current_generator=generator_state,
        fingerprint=loader._fingerprint,
    ).to_dict()


def restore_state(loader: Any, payload: dict[str, object]) -> None:
    """Install compat progress and restore its ambient globals immediately."""
    state = CompatZeroCheckpoint.from_dict(payload)
    require_fingerprint_match(state.fingerprint, loader._fingerprint)
    loader.close()
    restore_generator(loader._compat_generator, state.current_generator)
    restore_globals(state.current_globals)
    loader._resume_compat_state = state


def _torch_loader(loader: Any) -> Any:
    options = {
        "dataset": loader.dataset,
        "batch_size": 1 if loader.batch_sampler is not None else loader.batch_size,
        "shuffle": loader.shuffle,
        "sampler": loader.sampler,
        "batch_sampler": loader.batch_sampler,
        "num_workers": 0,
        "collate_fn": loader.collate_fn,
        "pin_memory": loader.pin_memory,
        "drop_last": loader.drop_last,
        "timeout": loader.timeout,
        "worker_init_fn": loader.worker_init_fn,
        "multiprocessing_context": loader.multiprocessing_context,
        "generator": loader._compat_generator,
        "prefetch_factor": None,
        "persistent_workers": False,
        "pin_memory_device": loader.pin_memory_device,
    }
    if "in_order" in inspect.signature(torch.utils.data.DataLoader).parameters:
        options["in_order"] = loader.in_order
    return torch.utils.data.DataLoader(**options)


def _start_torch_iterator(loader: Any) -> Iterator[Any]:
    """Start torch's iterator at its observable eager base-seed boundary."""
    return iter(loader._compat_loader)

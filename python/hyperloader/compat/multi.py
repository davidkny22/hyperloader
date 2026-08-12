"""Torch-compatible execution across real operating-system worker lanes."""

from __future__ import annotations

import weakref
from typing import Any

import torch

from hyperloader.config import AUTO
from hyperloader.epoch import EpochState
from hyperloader.fingerprint import (
    build_contract_fingerprint,
    build_dataset_fingerprint,
    require_fingerprint_match,
)

from .multi_iterator import CompatMultiIterator
from .multi_loader import reference_loader, worker_loader
from .multi_state import CompatMultiCheckpoint
from .rng import capture_torch_source, clone_torch_source, restore_torch_source


def prepare(loader: Any, workers: int) -> None:
    """Prepare torch's worker semantics without starting a process at construction."""
    if workers <= 0:
        raise ValueError("multi-worker compatibility requires positive num_workers")
    loader.num_workers = workers
    if not loader._persistent_workers_explicit:
        loader.persistent_workers = False
    loader.prefetch_factor = (
        None if loader.prefetch_factor is AUTO else loader.prefetch_factor
    )
    generator = loader.generator
    if generator is None and loader.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(loader.seed)
    loader._compat_generator = generator
    loader.root_seed = 0 if generator is None else int(generator.initial_seed())
    loader._epoch_state = EpochState()
    loader._resume_compat_state = None
    loader._resume_compat_multi_state: CompatMultiCheckpoint | None = None
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
    loader._compat_reference = reference_loader(loader)
    loader._compat_loader = None
    loader._compat_has_started = False
    loader._fingerprint = build_contract_fingerprint(loader)


def iterate(loader: Any) -> CompatMultiIterator:
    """Create one worker iterator or restore a strict delivered prefix."""
    active = (
        None if loader._active_iterator_ref is None else loader._active_iterator_ref()
    )
    if active is not None and not active.complete:
        retain_workers = loader.persistent_workers and loader._compat_loader is not None
        active.invalidate(shutdown=not retain_workers)
        if not retain_workers:
            loader._compat_loader = None
    resume = loader._resume_compat_multi_state
    if resume is None:
        reused_base_seed = bool(
            loader._compat_has_started
            and loader.persistent_workers
            and loader._compat_loader is not None
        )
        if loader._compat_loader is None:
            loader._compat_loader = (
                worker_loader(loader, 0, 0, {}, {})
                if loader.config.determinism.compat_resume == "on"
                else loader._compat_reference
            )
        iterator_generator = capture_torch_source(loader._compat_generator)
        wrapper = CompatMultiIterator(
            loader,
            iter(loader._compat_loader),
            iterator_generator,
            tagged=loader.config.determinism.compat_resume == "on",
            reused_base_seed=reused_base_seed,
        )
    else:
        base_generator = (
            clone_torch_source(resume.iterator_generator)
            if resume.reused_base_seed
            else None
        )
        loader._compat_loader = worker_loader(
            loader,
            resume.delivered_batches,
            resume.assignment_phase,
            resume.lane_states,
            resume.lane_seeds,
            base_generator=base_generator,
        )
        restore_torch_source(loader._compat_generator, resume.iterator_generator)
        wrapper = CompatMultiIterator(
            loader,
            iter(loader._compat_loader),
            resume.iterator_generator,
            delivered=resume.delivered_batches,
            dummy_batches=resume.assignment_phase,
            reused_base_seed=resume.reused_base_seed,
        )
        wrapper.prime_resume()
        restore_torch_source(loader._compat_generator, resume.current_generator)
    loader._resume_compat_multi_state = None
    loader._compat_has_started = True
    loader._active_iterator_ref = weakref.ref(wrapper)
    return wrapper


def capture_state(loader: Any) -> dict[str, object]:
    """Capture an opt-in same-width worker continuation."""
    if loader.config.determinism.compat_resume != "on":
        raise RuntimeError(
            "multi-worker compat state requires determinism.compat_resume='on'"
        )
    _require_stateless_map_resume(loader)
    active = (
        None if loader._active_iterator_ref is None else loader._active_iterator_ref()
    )
    if active is not None and not active.complete:
        return active.capture_checkpoint().to_dict()
    if loader._resume_compat_multi_state is not None:
        return loader._resume_compat_multi_state.to_dict()
    state = capture_torch_source(loader._compat_generator)
    return CompatMultiCheckpoint(
        delivered_batches=0,
        worker_count=loader.num_workers,
        assignment_phase=0,
        reused_base_seed=False,
        iterator_generator=state,
        current_generator=state,
        lane_states={},
        lane_seeds={},
        fingerprint=loader._fingerprint,
    ).to_dict()


def restore_state(loader: Any, payload: dict[str, object]) -> None:
    """Install a validated same-width worker continuation."""
    if loader.config.determinism.compat_resume != "on":
        raise RuntimeError(
            "multi-worker compat state requires determinism.compat_resume='on'"
        )
    _require_stateless_map_resume(loader)
    state = CompatMultiCheckpoint.from_dict(payload)
    if state.worker_count != loader.num_workers:
        raise ValueError("compat checkpoint requires the same num_workers")
    require_fingerprint_match(state.fingerprint, loader._fingerprint)
    active = (
        None if loader._active_iterator_ref is None else loader._active_iterator_ref()
    )
    if active is not None:
        active.invalidate()
    loader._active_iterator_ref = None
    loader._compat_loader = None
    restore_torch_source(loader._compat_generator, state.current_generator)
    loader._resume_compat_multi_state = state


def _require_stateless_map_resume(loader: Any) -> None:
    if not loader.in_order:
        raise RuntimeError("multi-worker compat resume requires in_order=True")
    if isinstance(loader.dataset, torch.utils.data.IterableDataset):
        raise RuntimeError(  # noqa: TRY004
            "multi-worker compat resume requires a map-style dataset without "
            "cross-sample internal state"
        )

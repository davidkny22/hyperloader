"""Torch DataLoader construction for compatibility worker lanes."""

from __future__ import annotations

import inspect
from typing import Any

import torch

from .collation import CollateAdapter
from .dataset import IterableDatasetAdapter, MapDatasetAdapter
from .sampling import BatchSamplerAdapter, SamplerAdapter
from .worker import WorkerInitializer


def reference_loader(loader: Any) -> Any:
    """Build the exact uninstrumented torch worker path."""
    options = _common_options(loader)
    options.update(
        dataset=loader.dataset,
        batch_size=1 if loader.batch_sampler is not None else loader.batch_size,
        shuffle=loader.shuffle,
        sampler=loader.sampler,
        batch_sampler=loader.batch_sampler,
        collate_fn=loader.collate_fn,
        drop_last=loader.drop_last,
        worker_init_fn=loader.worker_init_fn,
    )
    return torch.utils.data.DataLoader(**options)


def worker_loader(
    loader: Any,
    skip: int,
    phase: int,
    lane_states: dict[int, bytes],
    lane_seeds: dict[int, int],
    *,
    base_generator: Any = None,
) -> Any:
    """Build the tagged worker path used by opt-in continuation."""
    reference = loader._compat_reference
    iterable = isinstance(loader.dataset, torch.utils.data.IterableDataset)
    adapter = (
        IterableDatasetAdapter(loader.dataset, loader.batch_size, loader.num_workers)
        if iterable
        else MapDatasetAdapter(loader.dataset)
    )
    auto_collation = reference.batch_sampler is not None
    options = _common_options(loader, generator=base_generator)
    options.update(
        dataset=adapter,
        collate_fn=CollateAdapter(
            loader.collate_fn,
            auto_collation=auto_collation,
            pin_memory_device=loader.pin_memory_device,
        ),
        worker_init_fn=WorkerInitializer(
            loader.worker_init_fn,
            lane_states,
            lane_seeds,
        ),
    )
    if iterable:
        options.update(
            batch_size=loader.batch_size,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            drop_last=loader.drop_last,
        )
    elif auto_collation:
        options.update(
            batch_size=1,
            shuffle=False,
            sampler=None,
            batch_sampler=BatchSamplerAdapter(
                reference.batch_sampler,
                skip=skip,
                phase=phase,
            ),
            drop_last=False,
        )
    else:
        options.update(
            batch_size=None,
            shuffle=False,
            sampler=SamplerAdapter(reference.sampler, skip=skip, phase=phase),
            batch_sampler=None,
            drop_last=False,
        )
    return torch.utils.data.DataLoader(**options)


def _common_options(loader: Any, *, generator: Any = None) -> dict[str, Any]:
    options = {
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
        "timeout": loader.timeout,
        "multiprocessing_context": loader.multiprocessing_context,
        "generator": loader._compat_generator if generator is None else generator,
        "prefetch_factor": loader.prefetch_factor,
        "persistent_workers": loader.persistent_workers,
        "pin_memory_device": loader.pin_memory_device,
    }
    if "in_order" in inspect.signature(torch.utils.data.DataLoader).parameters:
        options["in_order"] = loader.in_order
    return options

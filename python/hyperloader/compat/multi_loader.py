"""Torch DataLoader construction for compatibility worker lanes."""

from __future__ import annotations

import inspect
from typing import Any

import torch


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


def _common_options(loader: Any) -> dict[str, Any]:
    options = {
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
        "timeout": loader.timeout,
        "multiprocessing_context": loader.multiprocessing_context,
        "generator": loader._compat_generator,
        "prefetch_factor": loader.prefetch_factor,
        "persistent_workers": loader.persistent_workers,
        "pin_memory_device": loader.pin_memory_device,
    }
    if "in_order" in inspect.signature(torch.utils.data.DataLoader).parameters:
        options["in_order"] = loader.in_order
    return options

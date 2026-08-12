"""Torch collation wrapped with compat lane restore metadata."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data._utils.collate import default_collate, default_convert

from .protocol import TaggedBatch, TaggedSample


class CollateAdapter:
    """Apply torch's real collation and retain the pre-fetch lane state."""

    def __init__(
        self,
        collate_fn: Any,
        *,
        auto_collation: bool,
        pin_memory_device: str,
    ) -> None:
        self.collate_fn = collate_fn or (
            default_collate if auto_collation else default_convert
        )
        self.pin_memory_device = pin_memory_device or None

    def __call__(self, samples: Any) -> TaggedBatch:
        items = samples if isinstance(samples, list) else [samples]
        if not items or not all(isinstance(item, TaggedSample) for item in items):
            raise RuntimeError("compat collation received an untagged sample")
        first = items[0]
        state = next((item.state for item in items if item.state is not None), None)
        if state is None:
            raise RuntimeError("compat batch has no pre-fetch worker state")
        info = torch.utils.data.get_worker_info()
        if info is None:
            raise RuntimeError("compat collation requires a worker process")
        value = None if first.dummy else self.collate_fn(
            [item.value for item in items] if isinstance(samples, list) else first.value
        )
        return TaggedBatch(
            first.batch,
            info.id,
            info.seed,
            value,
            state,
            first.dummy,
            self.pin_memory_device,
        )

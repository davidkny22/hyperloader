"""Torch-compatible pinned-memory delivery owned by the lane pool."""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import torch
from torch.utils.data._utils.pin_memory import pin_memory

from .protocol import TaggedBatch


class CompatPinning:
    """Resolve Torch's pin-memory regime once per operating-system pool."""

    def __init__(self, loader: Any) -> None:
        accelerator = getattr(torch, "accelerator", None)
        available = bool(accelerator is not None and accelerator.is_available())
        if loader.pin_memory and loader.pin_memory_device:
            warnings.warn(
                "pin_memory_device is deprecated, the current accelerator will be "
                f"used as the device,ignore pin_memory_device='{loader.pin_memory_device}'.",
                stacklevel=3,
            )
        if loader.pin_memory and not available:
            warnings.warn(
                "'pin_memory' argument is set as true but no accelerator is found, "
                "then device pinned memory won't be used.",
                stacklevel=3,
            )
        self._enabled = bool(loader.pin_memory and available)
        current = accelerator.current_accelerator() if self._enabled else None
        self._device = None if current is None else current.type
        if self._device == "mps":
            self._enabled = False
            warnings.warn(
                "'pin_memory' argument is set as true but not supported on MPS now, "
                "device pinned memory won't be used.",
                stacklevel=3,
            )

    def pin(self, value: Any) -> Any:
        """Pin one decoded batch while preserving checkpoint envelopes."""
        if not self._enabled:
            return value
        if isinstance(value, TaggedBatch):
            return dataclasses.replace(
                value,
                value=pin_memory(value.value, self._device),
            )
        return pin_memory(value, self._device)

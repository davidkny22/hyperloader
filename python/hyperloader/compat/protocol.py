"""Pickle-safe envelopes for compat worker transport."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TaggedIndex:
    """Carry a batch coordinate through torch's worker index queue."""

    batch: int
    offset: int
    index: Any
    dummy: bool = False


@dataclass(frozen=True, slots=True)
class TaggedSample:
    """Carry one user value and the state before its batch fetch."""

    batch: int
    value: Any
    state: bytes | None
    dummy: bool = False


@dataclass(frozen=True, slots=True)
class TaggedBatch:
    """Carry one collated value and its worker restore point to the owner."""

    batch: int
    worker: int
    seed: int
    value: Any
    state: bytes
    dummy: bool = False
    pin_memory_device: str | None = None

    def pin_memory(self) -> TaggedBatch:
        """Preserve the envelope while torch pins the actual user value."""
        from torch.utils.data._utils.pin_memory import pin_memory

        return dataclasses.replace(
            self,
            value=pin_memory(self.value, self.pin_memory_device),
        )

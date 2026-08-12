"""Pickle-safe envelopes for compat worker transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TaggedBatch:
    """Carry one collated value and its worker restore point to the owner."""

    batch: int
    worker: int
    seed: int
    value: Any
    state: bytes


@dataclass(frozen=True, slots=True)
class LaneExhausted:
    """Mark one iterable worker lane as exhausted without raising an error."""

    worker: int

"""Immutable fallback dispatch message."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkerCommand:
    """One immutable dispatch decoded by a fallback worker."""

    position: int
    epoch: int
    index: int
    stage_plan: int
    worker: int
    batch_len: int
    slot: int

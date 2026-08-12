"""State owned by one logical iterable lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class IterableLane:
    """Retain one lane-owned dataset copy and live source iterator."""

    identity: int
    dataset: Any
    iterator: Any
    arrival: int = 0
    produced_batches: int = 0

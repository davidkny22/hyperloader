"""Shared current-sample pointer for lazily armed RNG surfaces."""

from __future__ import annotations

SampleRng = tuple[int, int, int]
TORCH_SEED = 0
KEY = 1
COORD = 2


class CurrentSample:
    """Expose one pointer that changes at each sample boundary."""

    def __init__(self) -> None:
        self.value: SampleRng | None = None

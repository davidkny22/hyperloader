"""Shared current-sample pointer for lazily armed RNG surfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SampleRng:
    """Carry every value needed to arm one sample's seeded surfaces."""

    torch_seed: int
    key: int
    coord: int


class CurrentSample:
    """Expose one pointer that changes at each sample boundary."""

    def __init__(self) -> None:
        self.value: SampleRng | None = None

    def update(self, torch_seed: int, key: int, coord: int) -> SampleRng:
        """Publish one immutable sample token with one pointer assignment."""
        sample = SampleRng(torch_seed, key, coord)
        self.value = sample
        return sample

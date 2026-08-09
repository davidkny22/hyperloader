"""Map-style black-box planning and native sampler coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hyperloader import _hyperloader


@dataclass(frozen=True, slots=True)
class BlackBoxPlan:
    """Treat the dataset's complete item access as one process stage."""

    length: int
    shuffle: bool

    def index(self, root_seed: int, epoch: int, position: int) -> int:
        """Map one sampler position to its dataset index."""
        if not 0 <= position < self.length:
            raise IndexError("sampler position is outside the dataset")
        if not self.shuffle:
            return position
        return _hyperloader._permutation_index(
            root_seed, epoch, self.length, position
        )


def build_black_box_plan(dataset: Any, shuffle: bool | None) -> BlackBoxPlan | None:
    """Plan a sized map-style dataset or defer unsupported iterable planning."""
    try:
        length = len(dataset)
    except TypeError:
        return None
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError("dataset length must be a nonnegative integer")
    return BlackBoxPlan(length=length, shuffle=bool(shuffle))

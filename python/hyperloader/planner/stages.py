"""Planning contract for an explicit typed stage pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from hyperloader import _hyperloader
from hyperloader.stages import Pipeline


@dataclass(frozen=True, slots=True)
class StagePlan:
    """Route one validated stage pipeline without decomposing user code."""

    length: int
    shuffle: bool
    sample_thread_safe: bool

    def index(self, root_seed: int, epoch: int, position: int) -> int:
        """Map one sampler position to its source index."""
        if not 0 <= position < self.length:
            raise IndexError("sampler position is outside the pipeline source")
        if not self.shuffle:
            return position
        return _hyperloader._permutation_index(root_seed, epoch, self.length, position)


def build_stage_plan(
    dataset: Pipeline[object, object], shuffle: bool | None
) -> StagePlan:
    """Build an executable plan from a validated public pipeline."""
    return StagePlan(
        length=len(dataset),
        shuffle=bool(shuffle),
        sample_thread_safe=dataset.sample_thread_safe,
    )

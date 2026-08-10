"""Shared plan representation for structure-decomposed datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hyperloader import _hyperloader
from hyperloader.stages import StageIO, ThreadSafety


@dataclass(frozen=True, slots=True)
class StructureStage:
    """Describe one co-resident operation selected by a planner mapping."""

    name: str
    io: StageIO = StageIO.NONE
    thread_safety: ThreadSafety = ThreadSafety.ISOLATED
    cost_hint_ns: int | None = None


@dataclass(frozen=True, slots=True)
class StructurePlan:
    """Route a validated dataset adapter through the standard sampler contract."""

    length: int
    shuffle: bool
    mapping_id: str
    stages: tuple[StructureStage, ...]
    execution_dataset: Any = field(repr=False, compare=False)

    def index(self, root_seed: int, epoch: int, position: int) -> int:
        """Map one sampler position to its structure-adapter index."""
        if not 0 <= position < self.length:
            raise IndexError("sampler position is outside the structured dataset")
        if not self.shuffle:
            return position
        return _hyperloader._permutation_index(root_seed, epoch, self.length, position)

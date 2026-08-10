"""Structure decomposition for torchvision folder datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hyperloader.stages import StageIO

from .plan import StructurePlan, StructureStage


@dataclass(frozen=True, slots=True)
class FolderAdapter:
    """Execute DatasetFolder's source, decode, and transform sequence exactly."""

    dataset: Any

    @property
    def worker_dataset(self) -> Any:
        """Expose the user's dataset through get_worker_info()."""
        return self.dataset

    def __len__(self) -> int:
        return len(self.dataset.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        path, target = self.dataset.samples[index]
        sample = self.dataset.loader(path)
        if self.dataset.transform is not None:
            sample = self.dataset.transform(sample)
        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        return sample, target


def build_plan(dataset: Any, shuffle: bool | None) -> StructurePlan | None:
    """Build a folder plan only when the canonical structure is present."""
    required = ("samples", "loader", "transform", "target_transform")
    if not all(hasattr(dataset, name) for name in required):
        return None
    adapter = FolderAdapter(dataset)
    return StructurePlan(
        length=len(adapter),
        shuffle=bool(shuffle),
        mapping_id="torchvision-dataset-folder",
        stages=(
            StructureStage("sample-path", io=StageIO.READ),
            StructureStage("loader-decode"),
            StructureStage("sample-transform"),
            StructureStage("target-transform"),
        ),
        execution_dataset=adapter,
    )

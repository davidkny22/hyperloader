"""Structure decomposition for torch TensorDataset values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .plan import StructurePlan, StructureStage


@dataclass(frozen=True, slots=True)
class TensorDatasetAdapter:
    """Index tensor storage directly without invoking a Python dataset method."""

    dataset: Any

    @property
    def worker_dataset(self) -> Any:
        """Expose the user's dataset through get_worker_info()."""
        return self.dataset

    def __len__(self) -> int:
        return self.dataset.tensors[0].size(0)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        return tuple(tensor[index] for tensor in self.dataset.tensors)

    def native_batch(self, start: int, stop: int) -> list[Any]:
        """Return default-collated tensor views for one contiguous range."""
        return [tensor[start:stop] for tensor in self.dataset.tensors]


def build_plan(dataset: Any, shuffle: bool | None) -> StructurePlan | None:
    """Build a tensor-tuple plan when every leading dimension agrees."""
    tensors = getattr(dataset, "tensors", None)
    if not isinstance(tensors, tuple) or not tensors:
        return None
    length = tensors[0].size(0)
    if any(tensor.size(0) != length for tensor in tensors):
        return None
    return StructurePlan(
        length=length,
        shuffle=bool(shuffle),
        mapping_id="torch-tensor-dataset",
        stages=(StructureStage("tensor-row-view"),),
        execution_dataset=TensorDatasetAdapter(dataset),
        native_batch=not bool(shuffle),
    )

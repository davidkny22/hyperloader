"""Structure decomposition for Hugging Face Arrow datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hyperloader.stages import StageIO

from .plan import StructurePlan, StructureStage


@dataclass(frozen=True, slots=True)
class ArrowDatasetAdapter:
    """Execute Arrow query and configured formatting as co-resident operations."""

    dataset: Any

    @property
    def worker_dataset(self) -> Any:
        """Expose the user's dataset through get_worker_info()."""
        return self.dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        from datasets.formatting import format_table, get_formatter, query_table

        format_kwargs = self.dataset._format_kwargs or {}
        formatter = get_formatter(
            self.dataset._format_type,
            features=self.dataset._info.features,
            **format_kwargs,
        )
        table = query_table(self.dataset._data, index, indices=self.dataset._indices)
        return format_table(
            table,
            index,
            formatter=formatter,
            format_columns=self.dataset._format_columns,
            output_all_columns=self.dataset._output_all_columns,
        )


def build_plan(dataset: Any, shuffle: bool | None) -> StructurePlan | None:
    """Build an Arrow plan only when the formatter state is recognizable."""
    required = (
        "_data",
        "_indices",
        "_format_type",
        "_format_columns",
        "_output_all_columns",
        "_format_kwargs",
        "_info",
    )
    if not all(hasattr(dataset, name) for name in required):
        return None
    adapter = ArrowDatasetAdapter(dataset)
    return StructurePlan(
        length=len(adapter),
        shuffle=bool(shuffle),
        mapping_id="huggingface-arrow-dataset",
        stages=(
            StructureStage("arrow-query", io=StageIO.READ),
            StructureStage("configured-format"),
        ),
        execution_dataset=adapter,
    )

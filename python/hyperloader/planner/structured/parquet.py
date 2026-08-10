"""Structure decomposition for local PyArrow Parquet datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hyperloader.stages import StageIO

from .plan import StructurePlan, StructureStage


@dataclass(slots=True)
class ParquetDatasetAdapter:
    """Reconstruct a local Parquet dataset and decode one logical row."""

    paths: tuple[str, ...]
    schema: Any
    length: int
    _dataset: Any = field(default=None, init=False, repr=False)

    @property
    def worker_dataset(self) -> Any:
        """Expose the lazy worker-local adapter through get_worker_info()."""
        return self

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.length:
            raise IndexError("Parquet row index is outside the dataset")
        return self._source().take([index]).to_pylist()[0]

    def _source(self) -> Any:
        if self._dataset is None:
            import pyarrow.dataset as ds

            self._dataset = ds.dataset(
                self.paths,
                format="parquet",
                schema=self.schema,
            )
        return self._dataset


def build_plan(dataset: Any, shuffle: bool | None) -> StructurePlan | None:
    """Build a local Parquet plan when partition reconstruction is unnecessary."""
    try:
        fragments = tuple(dataset.get_fragments())
    except (AttributeError, TypeError):
        return None
    if not fragments:
        return None
    if any(type(fragment).__name__ != "ParquetFileFragment" for fragment in fragments):
        return None
    if any(
        not dataset.schema.equals(fragment.physical_schema) for fragment in fragments
    ):
        return None
    paths = tuple(fragment.path for fragment in fragments)
    if any("://" in path for path in paths):
        return None
    adapter = ParquetDatasetAdapter(
        paths=paths,
        schema=dataset.schema,
        length=dataset.count_rows(),
    )
    return StructurePlan(
        length=len(adapter),
        shuffle=bool(shuffle),
        mapping_id="pyarrow-parquet-dataset",
        stages=(StructureStage("parquet-open-and-decode", io=StageIO.READ),),
        execution_dataset=adapter,
    )

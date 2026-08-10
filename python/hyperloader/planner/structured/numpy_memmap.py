"""Structure decomposition for NumPy memory-mapped arrays."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from hyperloader.stages import StageIO

from .plan import StructurePlan, StructureStage


@dataclass(slots=True)
class MemmapAdapter:
    """Reopen one memory map lazily inside each spawned worker."""

    filename: str
    dtype: Any
    mode: str
    offset: int
    shape: tuple[int, ...]
    order: str
    _mapped: Any = field(default=None, init=False, repr=False)

    @property
    def worker_dataset(self) -> Any:
        """Expose the lazy worker-local adapter through get_worker_info()."""
        return self

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: int) -> Any:
        import numpy as np

        value = self._array()[index]
        if isinstance(value, np.memmap):
            return value.view(np.ndarray)
        return value

    def _array(self) -> Any:
        if self._mapped is None:
            import numpy as np

            self._mapped = np.memmap(
                self.filename,
                dtype=self.dtype,
                mode=self.mode,
                offset=self.offset,
                shape=self.shape,
                order=self.order,
            )
        return self._mapped


def build_plan(dataset: Any, shuffle: bool | None) -> StructurePlan | None:
    """Build a reopenable map only for nonempty row-addressable arrays."""
    filename = getattr(dataset, "filename", None)
    shape = tuple(getattr(dataset, "shape", ()))
    if not isinstance(filename, (str, bytes, os.PathLike)) or not shape:
        return None
    order = (
        "F" if dataset.flags.f_contiguous and not dataset.flags.c_contiguous else "C"
    )
    adapter = MemmapAdapter(
        filename=os.fsdecode(filename),
        dtype=dataset.dtype,
        mode="r+" if dataset.mode == "w+" else dataset.mode,
        offset=dataset.offset,
        shape=shape,
        order=order,
    )
    return StructurePlan(
        length=len(adapter),
        shuffle=bool(shuffle),
        mapping_id="numpy-memmap",
        stages=(StructureStage("memmap-row-read", io=StageIO.READ),),
        execution_dataset=adapter,
    )

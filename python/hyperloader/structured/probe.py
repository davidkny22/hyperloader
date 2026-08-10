"""Construction probe retained for batch-native delivery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..planner import StructurePlan
from ..process.shape import batch_shape


@dataclass(slots=True)
class NativeBatchProbe:
    """Retain the first batch or its deferred failure for exact first delivery."""

    value: Any = None
    error: BaseException | None = None


def is_native_batch_path(loader: Any) -> bool:
    """Return whether the loader configuration admits owner-side batch execution."""
    plan_enabled = isinstance(loader._plan, StructurePlan) and loader._plan.native_batch
    adapter_enabled = bool(
        getattr(loader._execution_dataset, "native_batch_enabled", False)
    )
    execution_enabled = adapter_enabled or not loader._sample_thread_safe
    return bool(
        (plan_enabled or adapter_enabled)
        and loader.batch_size is not None
        and isinstance(loader.num_workers, int)
        and loader.num_workers > 0
        and loader.sampler is None
        and loader.batch_sampler is None
        and loader.collate_fn is None
        and loader.mode == "native"
        and execution_enabled
    )


def prepare_native_batch(loader: Any) -> None:
    """Execute and retain the first batch once so shape probing is not a reread."""
    if not is_native_batch_path(loader) or loader._plan.length == 0:
        return
    length = loader._plan.length
    if loader.drop_last:
        length -= length % loader.batch_size
    if length == 0:
        return
    stop = min(loader.batch_size, length)
    try:
        value = loader._execution_dataset.native_batch(0, stop)
    except BaseException as error:
        loader._native_batch_probe = NativeBatchProbe(error=error)
        return
    loader._native_batch_probe = NativeBatchProbe(value=value)
    loader._native_batch_shape = {**batch_shape(value, None), "source": "probe"}

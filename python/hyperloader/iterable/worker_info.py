"""Scoped torch worker identity for sequential logical lanes."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def lane_worker_info(
    lane: int,
    lane_count: int,
    dataset: Any,
    seed: int | None,
) -> Iterator[None]:
    """Publish one immutable logical-lane identity around user code."""
    from torch.utils.data._utils import worker as worker_module

    prior = worker_module._worker_info
    worker_module._worker_info = worker_module.WorkerInfo(
        id=lane,
        num_workers=lane_count,
        seed=seed,
        dataset=dataset,
    )
    try:
        yield
    finally:
        worker_module._worker_info = prior

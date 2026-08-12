"""Scoped torch worker identity for sequential logical lanes."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from hyperloader.process.worker_info import WorkerInfoContext


@contextmanager
def lane_worker_info(
    lane: int,
    lane_count: int,
    dataset: Any,
    seed: int | None,
) -> Iterator[None]:
    """Publish one immutable logical-lane identity around user code."""
    context = WorkerInfoContext(lane, lane_count, dataset)
    context.begin_sample(seed)
    try:
        yield
    finally:
        context.clear()

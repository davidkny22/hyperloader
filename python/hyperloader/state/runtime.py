"""Iterator coordinate helpers for map-style resume."""

from __future__ import annotations

from typing import Any


def resume_sample_position(loader: Any, length: int) -> int:
    """Resolve the one-shot delivered-batch cursor to a sample position."""
    cursor = loader._resume_cursor_batches
    batch_size = loader.batch_size or 1
    total_batches = (length + batch_size - 1) // batch_size
    if cursor > total_batches:
        raise ValueError(
            f"loader state cursor {cursor} exceeds {total_batches} delivered batches"
        )
    return min(length, cursor * batch_size)

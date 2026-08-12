"""Logical-lane execution for iterable datasets."""

from .factory import logical_lane_count, prepare_iterable_source
from .iterator import IterableIterator

__all__ = ["IterableIterator", "logical_lane_count", "prepare_iterable_source"]

"""Logical-lane execution for iterable datasets."""

from .factory import logical_lane_count, prepare_iterable_source
from .iterator import IterableIterator
from .rng import IterableRngSession, iterable_coordinate
from .state import capture_iterable_state, restore_iterable_state

__all__ = [
    "IterableIterator",
    "IterableRngSession",
    "capture_iterable_state",
    "iterable_coordinate",
    "logical_lane_count",
    "prepare_iterable_source",
    "restore_iterable_state",
]

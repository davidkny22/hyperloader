"""Construction-time iterable payload retention."""

from __future__ import annotations

import pickle
from typing import Any


def logical_lane_count(loader: Any) -> int:
    """Return the fixed logical lane count for an iterable loader."""
    workers = loader.num_workers
    return workers if isinstance(workers, int) and workers > 0 else 1


def prepare_iterable_source(loader: Any) -> bytes | None:
    """Retain one payload that can instantiate every logical lane."""
    lanes = logical_lane_count(loader)
    try:
        payload = pickle.dumps(loader.dataset, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.loads(payload)
    except Exception as error:
        if lanes == 1:
            return None
        raise TypeError(
            "iterable logical lanes require a dataset that survives a pickle "
            "round-trip; set num_workers=0 for the L=1 fallback"
        ) from error
    return payload

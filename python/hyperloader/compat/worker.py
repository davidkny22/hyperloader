"""Worker initialization and free-running RNG state for compat lanes."""

from __future__ import annotations

import pickle

from .rng import capture_globals, restore_globals


def capture_worker_state() -> bytes:
    """Serialize the three free-running CPU RNG states in one lane."""
    return pickle.dumps(capture_globals(), protocol=pickle.HIGHEST_PROTOCOL)


def restore_worker_state(payload: bytes) -> None:
    """Restore one validated lane RNG state bundle."""
    restore_globals(pickle.loads(payload))

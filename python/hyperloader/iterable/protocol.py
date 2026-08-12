"""Stateful iterable source protocol validation and calls."""

from __future__ import annotations

from typing import Any


def has_state_pair(dataset: Any) -> bool:
    """Return whether a source supplies both callable state methods."""
    save = getattr(dataset, "state_dict", None)
    load = getattr(dataset, "load_state_dict", None)
    if save is None and load is None:
        return False
    if not callable(save) or not callable(load):
        raise TypeError(
            "iterable stateful source requires callable state_dict and load_state_dict"
        )
    return True


def capture_source_state(dataset: Any) -> dict[str, Any]:
    """Capture one validated source state at a production boundary."""
    state = dataset.state_dict()
    if not isinstance(state, dict):
        raise TypeError("iterable source state_dict must return a dictionary")
    return state


def restore_source_state(dataset: Any, state: dict[str, Any]) -> None:
    """Install one previously captured source state."""
    dataset.load_state_dict(state)

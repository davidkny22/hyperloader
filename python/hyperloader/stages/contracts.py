"""Shared declarations for typed pipeline stages."""

from __future__ import annotations

from enum import Enum
from typing import Any


class StageIO(str, Enum):
    """Describe whether a stage performs external reads."""

    NONE = "none"
    READ = "read"


class ThreadSafety(str, Enum):
    """Describe the isolation promised by a Python stage."""

    ISOLATED = "isolated"
    THREAD_SAFE = "thread-safe"


def normalize_io(value: StageIO | str) -> StageIO:
    """Return a validated I/O declaration."""
    try:
        return StageIO(value)
    except ValueError as error:
        raise ValueError("io must be none or read") from error


def normalize_thread_safety(value: ThreadSafety | str) -> ThreadSafety:
    """Return a validated thread-safety declaration."""
    try:
        return ThreadSafety(value)
    except ValueError as error:
        raise ValueError("thread_safety must be isolated or thread-safe") from error


def validate_type(name: str, value: type[Any]) -> None:
    """Require a runtime type token for a stage edge."""
    if not isinstance(value, type):
        raise TypeError(f"{name} must be a type")


def validate_cost_hint(cost_hint_ns: int | None) -> None:
    """Require a positive nanosecond hint when one is supplied."""
    if cost_hint_ns is None:
        return
    if isinstance(cost_hint_ns, bool) or not isinstance(cost_hint_ns, int):
        raise TypeError("cost_hint_ns must be an integer or None")
    if cost_hint_ns <= 0:
        raise ValueError("cost_hint_ns must be positive")


def types_connect(output_type: type[Any], input_type: type[Any]) -> bool:
    """Return whether two declared stage edges can connect."""
    if output_type is object or input_type is object:
        return True
    return issubclass(output_type, input_type)

"""Typed source stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .contracts import (
    StageIO,
    ThreadSafety,
    normalize_io,
    normalize_thread_safety,
    validate_cost_hint,
    validate_type,
)

OutputT = TypeVar("OutputT")


@dataclass(frozen=True, slots=True)
class Source(Generic[OutputT]):
    """Read one map-style position from an indexable object or callable."""

    source: Any
    output_type: type[OutputT] = object
    io: StageIO | str = StageIO.READ
    thread_safety: ThreadSafety | str = ThreadSafety.ISOLATED
    cost_hint_ns: int | None = None
    length: int | None = None

    def __post_init__(self) -> None:
        validate_type("output_type", self.output_type)
        validate_cost_hint(self.cost_hint_ns)
        object.__setattr__(self, "io", normalize_io(self.io))
        object.__setattr__(
            self, "thread_safety", normalize_thread_safety(self.thread_safety)
        )
        length = self._resolved_length()
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ValueError("source length must be a nonnegative integer")

    def __len__(self) -> int:
        """Return the declared or provider-derived map length."""
        return self._resolved_length()

    def __call__(self, index: int) -> OutputT:
        """Read one source value."""
        if hasattr(self.source, "__getitem__"):
            return self.source[index]
        if callable(self.source):
            return self.source(index)
        raise TypeError("source must be indexable or callable")

    def _resolved_length(self) -> int:
        if self.length is not None:
            return self.length
        try:
            return len(self.source)
        except TypeError as error:
            raise TypeError("a callable source requires an explicit length") from error

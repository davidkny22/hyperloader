"""Typed transform stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from .contracts import (
    StageIO,
    ThreadSafety,
    normalize_io,
    normalize_thread_safety,
    validate_cost_hint,
    validate_type,
)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True, slots=True)
class Transform(Generic[InputT, OutputT]):
    """Apply one typed per-sample transformation."""

    fn: Callable[[InputT], OutputT]
    input_type: type[InputT] = object
    output_type: type[OutputT] = object
    io: StageIO | str = StageIO.NONE
    thread_safety: ThreadSafety | str = ThreadSafety.ISOLATED
    cost_hint_ns: int | None = None

    def __post_init__(self) -> None:
        if not callable(self.fn):
            raise TypeError("transform function must be callable")
        validate_type("input_type", self.input_type)
        validate_type("output_type", self.output_type)
        validate_cost_hint(self.cost_hint_ns)
        object.__setattr__(self, "io", normalize_io(self.io))
        object.__setattr__(
            self, "thread_safety", normalize_thread_safety(self.thread_safety)
        )

    def __call__(self, value: InputT) -> OutputT:
        """Transform one value."""
        return self.fn(value)

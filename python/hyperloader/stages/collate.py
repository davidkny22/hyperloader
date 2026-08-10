"""Typed collation stage."""

from __future__ import annotations

from collections.abc import Sequence
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
class Collate(Generic[InputT, OutputT]):
    """Combine delivered sample values into one typed batch."""

    fn: Callable[[Sequence[InputT]], OutputT]
    input_type: type[InputT] = object
    output_type: type[OutputT] = object
    io: StageIO | str = StageIO.NONE
    thread_safety: ThreadSafety | str = ThreadSafety.ISOLATED
    cost_hint_ns: int | None = None

    def __post_init__(self) -> None:
        if not callable(self.fn):
            raise TypeError("collate function must be callable")
        validate_type("input_type", self.input_type)
        validate_type("output_type", self.output_type)
        validate_cost_hint(self.cost_hint_ns)
        object.__setattr__(self, "io", normalize_io(self.io))
        object.__setattr__(
            self, "thread_safety", normalize_thread_safety(self.thread_safety)
        )

    def __call__(self, values: Sequence[InputT]) -> OutputT:
        """Collate one delivered sample sequence."""
        return self.fn(values)

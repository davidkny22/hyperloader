"""Validated composition of typed pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .collate import Collate
from .contracts import ThreadSafety, types_connect
from .decode import Decode
from .source import Source
from .transform import Transform

SampleT = TypeVar("SampleT")
BatchT = TypeVar("BatchT")
SampleStage = Decode[Any, Any] | Transform[Any, Any]


@dataclass(frozen=True, slots=True)
class Pipeline(Generic[SampleT, BatchT]):
    """Expose a typed stage chain as a map-style dataset."""

    source: Source[Any]
    sample_stages: tuple[SampleStage, ...]
    collate_stage: Collate[SampleT, BatchT]

    def __post_init__(self) -> None:
        previous = self.source.output_type
        decode_seen = False
        transform_seen = False
        for ordinal, stage in enumerate(self.sample_stages, start=1):
            if isinstance(stage, Decode):
                if decode_seen or transform_seen:
                    raise ValueError("Decode must appear at most once before Transform")
                decode_seen = True
            elif isinstance(stage, Transform):
                transform_seen = True
            else:
                raise TypeError(f"pipeline stage {ordinal} must be Decode or Transform")
            if not types_connect(previous, stage.input_type):
                raise TypeError(
                    f"pipeline type edge {previous.__name__} -> "
                    f"{stage.input_type.__name__} is incompatible"
                )
            previous = stage.output_type
        if not types_connect(previous, self.collate_stage.input_type):
            raise TypeError(
                f"pipeline type edge {previous.__name__} -> "
                f"{self.collate_stage.input_type.__name__} is incompatible"
            )

    def __len__(self) -> int:
        """Return the source map length."""
        return len(self.source)

    def __getitem__(self, index: int) -> SampleT:
        """Execute the co-resident per-sample chain for one source position."""
        value = self.source(index)
        for stage in self.sample_stages:
            value = stage(value)
        return value

    def collate(self, values: list[SampleT]) -> BatchT:
        """Execute the final batch stage over delivered sample views."""
        return self.collate_stage(values)

    @property
    def sample_thread_safe(self) -> bool:
        """Return whether every co-resident sample stage permits shared threads."""
        stages = (self.source, *self.sample_stages)
        return all(stage.thread_safety is ThreadSafety.THREAD_SAFE for stage in stages)

    @property
    def stages(self) -> tuple[object, ...]:
        """Return the immutable stage sequence in execution order."""
        return (self.source, *self.sample_stages, self.collate_stage)


def pipeline(*stages: object) -> Pipeline[Any, Any]:
    """Validate and compose Source, Decode, Transform, and Collate stages."""
    if len(stages) < 2:
        raise ValueError("pipeline requires Source and Collate stages")
    if not isinstance(stages[0], Source):
        raise TypeError("the first pipeline stage must be Source")
    if not isinstance(stages[-1], Collate):
        raise TypeError("the final pipeline stage must be Collate")
    return Pipeline(stages[0], tuple(stages[1:-1]), stages[-1])

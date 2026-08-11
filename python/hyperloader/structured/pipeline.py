"""Batch-native execution for exact library-owned pipeline shapes."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from hyperloader import _hyperloader

from ..decoder.execution import PinnedDecoder
from ..memory import ByteLedger
from ..stages import Decode, Pipeline
from .metrics import payload_bytes


@dataclass(slots=True)
class NativePipelineAdapter:
    """Run pinned decode or tokenized tensor stages in persistent native threads."""

    pipeline: Pipeline[Any, Any]
    worker_count: int
    source_class: str
    growth: str
    native_batch_enabled: bool = field(default=True, init=False)
    _executor: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)
    _futures: dict[int, Future[Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _epoch_length: int = field(default=0, init=False, repr=False)
    _next_submit: int = field(default=0, init=False, repr=False)
    _prefetch_depth: int = field(default=0, init=False, repr=False)
    _sample_shape: object = field(default=None, init=False, repr=False)
    _variable_shape: bool = field(default=False, init=False, repr=False)
    _produced_batches: int = field(default=0, init=False, repr=False)
    _produced_samples: int = field(default=0, init=False, repr=False)
    _sample_bytes: int = field(default=0, init=False, repr=False)
    _output_bytes: int = field(default=0, init=False, repr=False)
    _minimum_sample_bytes: int | None = field(default=None, init=False, repr=False)
    _maximum_sample_bytes: int = field(default=0, init=False, repr=False)
    _slot_capacity_bytes: int = field(default=0, init=False, repr=False)
    _arena: Any = field(default=None, init=False, repr=False)
    _closed_arena_stats: tuple[int, int, int, int] = field(
        default=(0, 0, 0, 0), init=False, repr=False
    )
    _memory: ByteLedger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._memory = ByteLedger(self.source_class, "single-write")

    def __len__(self) -> int:
        return len(self.pipeline)

    def native_batch(self, start: int, stop: int) -> Any:
        """Produce one final batch with no transport or reconstruction copy."""
        values = (
            list(self._pool().map(self.pipeline.__getitem__, range(start, stop)))
            if self._epoch_length == 0
            else [self._value(index) for index in range(start, stop)]
        )
        self._fill_frontier()
        value = self._collate_into_slot(values)
        self._record(values, value)
        return value

    def begin_native_epoch(self, length: int, depth: int, start: int) -> None:
        """Start a bounded decode frontier after any retained probe batch."""
        self._futures.clear()
        self._epoch_length = length
        self._next_submit = start
        self._prefetch_depth = max(1, depth)
        self._fill_frontier()

    def memory_report(self) -> dict[str, object]:
        """Return measured byte ownership for delivered native batches."""
        regions, growth_events, hold_events, overflow_events = self._arena_stats()
        return {
            **self._memory.report(),
            "loader_written_bytes": self._output_bytes,
            "maximum_sample_bytes": self._maximum_sample_bytes,
            "minimum_sample_bytes": self._minimum_sample_bytes or 0,
            "sample_output_bytes": self._sample_bytes,
            "growth_events": growth_events,
            "hold_events": hold_events,
            "overflow_events": overflow_events,
            "regions": regions,
            "slot_capacity_bytes": self._slot_capacity_bytes,
            "variable_shape": self._variable_shape,
        }

    def close(self) -> None:
        """Join native workers while permitting a later iterator to reopen them."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._futures.clear()
        self._epoch_length = 0
        self._next_submit = 0
        self._prefetch_depth = 0
        if self._arena is not None:
            self._closed_arena_stats = self._arena.stats()
            self._arena = None

    def _pool(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.worker_count,
                thread_name_prefix="hyperloader-native",
            )
        return self._executor

    def _value(self, index: int) -> Any:
        future = self._futures.pop(index, None)
        if future is None:
            future = self._pool().submit(self.pipeline.__getitem__, index)
        return future.result()

    def _fill_frontier(self) -> None:
        while (
            self._next_submit < self._epoch_length
            and len(self._futures) < self._prefetch_depth
        ):
            index = self._next_submit
            self._futures[index] = self._pool().submit(self.pipeline.__getitem__, index)
            self._next_submit += 1

    def _record(self, values: list[Any], batch: Any) -> None:
        output_size = payload_bytes(batch)
        sample_bytes = 0
        for value in values:
            signature = _shape_signature(value)
            if self._sample_shape is None:
                self._sample_shape = signature
            elif signature != self._sample_shape:
                self._variable_shape = True
            size = payload_bytes(value)
            sample_bytes += size
            self._sample_bytes += size
            self._maximum_sample_bytes = max(self._maximum_sample_bytes, size)
            self._minimum_sample_bytes = (
                size
                if self._minimum_sample_bytes is None
                else min(self._minimum_sample_bytes, size)
            )
        self._produced_batches += 1
        self._produced_samples += len(values)
        self._output_bytes += output_size
        self._memory.record(
            batch,
            len(values),
            pinned_stage_bytes=(
                sample_bytes if self.source_class == "pinned-decode" else 0
            ),
            arena_write_bytes=output_size,
        )

    def _collate_into_slot(self, values: list[Any]) -> Any:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        if not values:
            return self.pipeline.collate(values)
        first = values[0]
        if not isinstance(first, torch.Tensor) or first.device.type != "cpu":
            return self.pipeline.collate(values)
        if self.pipeline.collate_stage.fn is pad_sequence:
            return self._pad_into_slot(values)
        return self._stack_into_slot(values)

    def _stack_into_slot(self, values: list[Any]) -> Any:
        import torch

        first = values[0]
        shape = (len(values), *first.shape)
        output, slot, required = self._allocate_slot(shape, first)
        try:
            torch.stack(values, out=output)
            slot.publish(required)
            return output
        except BaseException:
            del output
            del slot
            raise

    def _pad_into_slot(self, values: list[Any]) -> Any:
        first = values[0]
        maximum = max(int(value.size(0)) for value in values)
        shape = (maximum, len(values), *first.shape[1:])
        output, slot, required = self._allocate_slot(shape, first)
        try:
            output.zero_()
            for column, value in enumerate(values):
                output[: value.size(0), column, ...].copy_(value)
            slot.publish(required)
            return output
        except BaseException:
            del output
            del slot
            raise

    def _allocate_slot(
        self, shape: tuple[int, ...], template: Any
    ) -> tuple[Any, Any, int]:
        import math
        import torch

        elements = math.prod(shape)
        required = elements * template.element_size()
        reserved = max(required, template.element_size())
        if self._arena is None:
            initial = _size_class(reserved, template.element_size())
            self._arena = _hyperloader._NativeArena(
                initial,
                max(2, self.worker_count),
                self.growth,
            )
        slot = self._arena.reserve(reserved)
        self._slot_capacity_bytes = max(self._slot_capacity_bytes, slot.capacity)
        output = torch.frombuffer(
            slot,
            dtype=template.dtype,
            count=elements,
        ).view(shape)
        return output, slot, required

    def _arena_stats(self) -> tuple[int, int, int, int]:
        return self._closed_arena_stats if self._arena is None else self._arena.stats()


def bind_native_pipeline(
    dataset: Any, *, shuffle: bool, worker_count: Any, growth: str
) -> Any:
    """Select an exact native adapter or retain the established pipeline refuge."""
    if (
        not isinstance(dataset, Pipeline)
        or shuffle
        or not isinstance(worker_count, int)
        or worker_count <= 0
        or not _known_collate(dataset.collate_stage.fn)
        or not _indexable_source(dataset)
    ):
        return dataset
    source_class = _source_class(dataset)
    if source_class is None:
        return dataset
    return NativePipelineAdapter(dataset, worker_count, source_class, growth)


def _source_class(dataset: Pipeline[Any, Any]) -> str | None:
    stages = dataset.sample_stages
    if (
        not stages
        and _declares_tensor(dataset.source.output_type)
        and _tensor_source(dataset.source.source)
    ):
        return "tokenized-text"
    if (
        len(stages) == 1
        and isinstance(stages[0], Decode)
        and isinstance(stages[0].fn, PinnedDecoder)
    ):
        return "pinned-decode"
    return None


def _known_collate(function: Any) -> bool:
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import default_collate

    return function in {torch.stack, pad_sequence, default_collate}


def _indexable_source(dataset: Pipeline[Any, Any]) -> bool:
    return type(dataset.source.source) in {list, tuple}


def _declares_tensor(value: type[Any]) -> bool:
    return value.__module__ == "torch" and value.__name__ == "Tensor"


def _tensor_source(source: list[Any] | tuple[Any, ...]) -> bool:
    if not source:
        return True
    first = source[0]
    first_type = type(first)
    return bool(
        first_type.__module__ == "torch"
        and first_type.__name__ == "Tensor"
        and first.device.type == "cpu"
        and not first.requires_grad
    )


def _shape_signature(value: Any) -> object:
    value_type = type(value)
    if value_type.__module__ == "torch" and value_type.__name__ == "Tensor":
        return (str(value.dtype), tuple(int(size) for size in value.shape))
    return value_type.__module__, value_type.__qualname__


def _size_class(size: int, element_size: int) -> int:
    minimum = max(size, element_size)
    return 1 << (minimum - 1).bit_length()

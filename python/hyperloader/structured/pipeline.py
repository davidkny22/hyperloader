"""Batch-native execution for exact library-owned pipeline shapes."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from ..decoder.execution import PinnedDecoder
from ..memory import ByteLedger
from ..stages import Decode, Pipeline
from .metrics import payload_bytes
from .native_slots import NativeSlotCollator


@dataclass(frozen=True, slots=True)
class _ProducedBatch:
    batch: Any
    samples: int
    sample_bytes: int
    output_bytes: int
    minimum_sample_bytes: int
    maximum_sample_bytes: int
    first_shape: object
    variable_shape: bool


@dataclass(slots=True)
class NativePipelineAdapter:
    """Run complete pinned batches in persistent native threads."""

    pipeline: Pipeline[Any, Any]
    worker_count: int
    source_class: str
    growth: str
    native_batch_enabled: bool = field(default=True, init=False)
    _sample_executor: ThreadPoolExecutor | None = field(
        default=None, init=False, repr=False
    )
    _batch_executor: ThreadPoolExecutor | None = field(
        default=None, init=False, repr=False
    )
    _futures: dict[int, Future[_ProducedBatch]] = field(
        default_factory=dict, init=False, repr=False
    )
    _epoch_length: int = field(default=0, init=False, repr=False)
    _next_batch_submit: int = field(default=0, init=False, repr=False)
    _batch_size: int = field(default=1, init=False, repr=False)
    _batch_prefetch: int = field(default=1, init=False, repr=False)
    _sample_shape: object = field(default=None, init=False, repr=False)
    _variable_shape: bool = field(default=False, init=False, repr=False)
    _produced_batches: int = field(default=0, init=False, repr=False)
    _produced_samples: int = field(default=0, init=False, repr=False)
    _sample_bytes: int = field(default=0, init=False, repr=False)
    _output_bytes: int = field(default=0, init=False, repr=False)
    _minimum_sample_bytes: int | None = field(default=None, init=False, repr=False)
    _maximum_sample_bytes: int = field(default=0, init=False, repr=False)
    _memory: ByteLedger = field(init=False, repr=False)
    _slots: NativeSlotCollator = field(init=False, repr=False)
    _record_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._memory = ByteLedger(self.source_class, "single-write")
        self._slots = NativeSlotCollator(self.worker_count, self.growth)

    def __len__(self) -> int:
        return len(self.pipeline)

    def native_batch(self, start: int, stop: int) -> Any:
        """Return one ordered batch whose final write ran off the owner thread."""
        if self._epoch_length == 0:
            values = list(
                self._sample_pool().map(self.pipeline.__getitem__, range(start, stop))
            )
            produced = self._finish_batch(values)
        else:
            future = self._futures.pop(start, None)
            if future is None:
                future = self._batch_pool().submit(self._produce_batch, start, stop)
            produced = future.result()
            self._fill_frontier()
        self._record(produced)
        return produced.batch

    def begin_native_epoch(
        self, length: int, depth: int, start: int, batch_size: int
    ) -> None:
        """Start a bounded complete-batch frontier after a retained probe."""
        if self._sample_executor is not None:
            self._sample_executor.shutdown(wait=True, cancel_futures=True)
            self._sample_executor = None
        self._futures.clear()
        self._epoch_length = length
        self._next_batch_submit = start
        self._batch_size = batch_size
        self._batch_prefetch = max(1, (depth + batch_size - 1) // batch_size)
        self._fill_frontier()

    def memory_report(self) -> dict[str, object]:
        """Return measured byte ownership for delivered native batches."""
        regions, growth_events, hold_events, overflow_events = self._slots.stats()
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
            "slot_capacity_bytes": self._slots.capacity_bytes,
            "variable_shape": self._variable_shape,
        }

    def enable_pinned_delivery(self) -> bool:
        """Route image collation directly into reusable pinned final slots."""
        if self.source_class != "pinned-decode":
            return False
        self._slots.enable_pinned()
        return True

    def close(self) -> None:
        """Join native workers while permitting a later iterator to reopen them."""
        if self._sample_executor is not None:
            self._sample_executor.shutdown(wait=True, cancel_futures=True)
            self._sample_executor = None
        if self._batch_executor is not None:
            self._batch_executor.shutdown(wait=True, cancel_futures=True)
            self._batch_executor = None
        self._futures.clear()
        self._epoch_length = 0
        self._next_batch_submit = 0
        self._batch_prefetch = 1
        self._slots.close()

    def _sample_pool(self) -> ThreadPoolExecutor:
        if self._sample_executor is None:
            self._sample_executor = ThreadPoolExecutor(
                max_workers=self.worker_count,
                thread_name_prefix="hyperloader-native-probe",
            )
        return self._sample_executor

    def _batch_pool(self) -> ThreadPoolExecutor:
        if self._batch_executor is None:
            self._batch_executor = ThreadPoolExecutor(
                max_workers=self.worker_count,
                thread_name_prefix="hyperloader-native-batch",
            )
        return self._batch_executor

    def _fill_frontier(self) -> None:
        while (
            self._next_batch_submit < self._epoch_length
            and len(self._futures) < self._batch_prefetch
        ):
            start = self._next_batch_submit
            stop = min(start + self._batch_size, self._epoch_length)
            self._futures[start] = self._batch_pool().submit(
                self._produce_batch, start, stop
            )
            self._next_batch_submit = stop

    def _produce_batch(self, start: int, stop: int) -> _ProducedBatch:
        values = [self.pipeline[index] for index in range(start, stop)]
        return self._finish_batch(values)

    def _finish_batch(self, values: list[Any]) -> _ProducedBatch:
        batch = self._slots.collate(values, self.pipeline.collate_stage.fn)
        shapes = [_shape_signature(value) for value in values]
        sizes = [payload_bytes(value) for value in values]
        return _ProducedBatch(
            batch=batch,
            samples=len(values),
            sample_bytes=sum(sizes),
            output_bytes=payload_bytes(batch),
            minimum_sample_bytes=min(sizes, default=0),
            maximum_sample_bytes=max(sizes, default=0),
            first_shape=shapes[0] if shapes else None,
            variable_shape=len(set(shapes)) > 1,
        )

    def _record(self, produced: _ProducedBatch) -> None:
        with self._record_lock:
            if self._sample_shape is None:
                self._sample_shape = produced.first_shape
            elif produced.first_shape != self._sample_shape:
                self._variable_shape = True
            self._variable_shape = self._variable_shape or produced.variable_shape
            self._sample_bytes += produced.sample_bytes
            self._maximum_sample_bytes = max(
                self._maximum_sample_bytes, produced.maximum_sample_bytes
            )
            if produced.samples:
                self._minimum_sample_bytes = (
                    produced.minimum_sample_bytes
                    if self._minimum_sample_bytes is None
                    else min(self._minimum_sample_bytes, produced.minimum_sample_bytes)
                )
            self._produced_batches += 1
            self._produced_samples += produced.samples
            self._output_bytes += produced.output_bytes
            self._memory.record(
                produced.batch,
                produced.samples,
                pinned_stage_bytes=(
                    produced.sample_bytes if self.source_class == "pinned-decode" else 0
                ),
                arena_write_bytes=produced.output_bytes,
            )


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

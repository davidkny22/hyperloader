"""Batch-native execution for exact library-owned pipeline shapes."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ..decoder.execution import PinnedDecoder
from ..stages import Decode, Pipeline
from .metrics import payload_bytes


@dataclass(slots=True)
class NativePipelineAdapter:
    """Run pinned decode or tokenized tensor stages in persistent native threads."""

    pipeline: Pipeline[Any, Any]
    worker_count: int
    source_class: str
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
        value = self.pipeline.collate(values)
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
        return {
            "bytes_beyond_irreducible": 0,
            "delivery": "single-write",
            "loader_written_bytes": self._output_bytes,
            "maximum_sample_bytes": self._maximum_sample_bytes,
            "minimum_sample_bytes": self._minimum_sample_bytes or 0,
            "sample_output_bytes": self._sample_bytes,
            "source_class": self.source_class,
            "produced_batches": self._produced_batches,
            "produced_samples": self._produced_samples,
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
        for value in values:
            signature = _shape_signature(value)
            if self._sample_shape is None:
                self._sample_shape = signature
            elif signature != self._sample_shape:
                self._variable_shape = True
            size = payload_bytes(value)
            self._sample_bytes += size
            self._maximum_sample_bytes = max(self._maximum_sample_bytes, size)
            self._minimum_sample_bytes = (
                size
                if self._minimum_sample_bytes is None
                else min(self._minimum_sample_bytes, size)
            )
        self._produced_batches += 1
        self._produced_samples += len(values)
        self._output_bytes += payload_bytes(batch)


def bind_native_pipeline(dataset: Any, *, shuffle: bool, worker_count: Any) -> Any:
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
    return NativePipelineAdapter(dataset, worker_count, source_class)


def _source_class(dataset: Pipeline[Any, Any]) -> str | None:
    stages = dataset.sample_stages
    if not stages and _declares_tensor(dataset.source.output_type):
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


def _shape_signature(value: Any) -> object:
    value_type = type(value)
    if value_type.__module__ == "torch" and value_type.__name__ == "Tensor":
        return (str(value.dtype), tuple(int(size) for size in value.shape))
    return value_type.__module__, value_type.__qualname__

"""Writable final-slot collation for batch-native pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from hyperloader import _hyperloader
from hyperloader.memory.pinned.pool import PinnedTensorPool


@dataclass(slots=True)
class NativeSlotCollator:
    """Reserve, fill, and publish refcounted native batch slots."""

    worker_count: int
    growth: str
    _arena: Any = field(default=None, init=False, repr=False)
    _closed_stats: tuple[int, int, int, int] = field(
        default=(0, 0, 0, 0), init=False, repr=False
    )
    _capacity_bytes: int = field(default=0, init=False, repr=False)
    _pinned_pool: PinnedTensorPool | None = field(default=None, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def collate(self, values: list[Any], fallback: Any) -> Any:
        """Collate tensors into one final slot or call the exact fallback."""
        import torch
        from torch.nn.utils.rnn import pad_sequence

        if not values:
            return fallback(values)
        first = values[0]
        if not isinstance(first, torch.Tensor) or first.device.type != "cpu":
            return fallback(values)
        if fallback is pad_sequence:
            return self._pad(values)
        return self._stack(values)

    def stats(self) -> tuple[int, int, int, int]:
        """Return live or final native arena counters."""
        if self._arena is None and self._pinned_pool is None:
            return self._closed_stats
        arena = (0, 0, 0, 0) if self._arena is None else self._arena.stats()
        if self._pinned_pool is None:
            return arena
        allocations = self._pinned_pool.allocation_count
        return (
            arena[0] + allocations,
            arena[1] + max(0, allocations - 1),
            arena[2] + self._pinned_pool.held_events,
            arena[3],
        )

    @property
    def capacity_bytes(self) -> int:
        """Return the largest reserved slot class."""
        pinned = 0 if self._pinned_pool is None else self._pinned_pool.capacity_bytes
        return max(self._capacity_bytes, pinned)

    def enable_pinned(self) -> None:
        """Make future final-slot writes target reusable pinned buffers."""
        if self._pinned_pool is None:
            self._pinned_pool = PinnedTensorPool()

    def close(self) -> None:
        """Release the arena owner while delivered views retain their slots."""
        with self._lock:
            self._closed_stats = self.stats()
            if self._arena is not None:
                self._arena = None
            if self._pinned_pool is not None:
                self._pinned_pool.close()
                self._pinned_pool = None

    def _stack(self, values: list[Any]) -> Any:
        import torch

        first = values[0]
        if self._pinned_pool is not None:
            output = self._allocate_pinned((len(values), *first.shape), first)
            torch.stack(values, out=output)
            return output
        output, slot, required = self._allocate((len(values), *first.shape), first)
        try:
            torch.stack(values, out=output)
            slot.publish(required)
            return output
        except BaseException:
            del output
            del slot
            raise

    def _pad(self, values: list[Any]) -> Any:
        first = values[0]
        maximum = max(int(value.size(0)) for value in values)
        shape = (maximum, len(values), *first.shape[1:])
        if self._pinned_pool is not None:
            output = self._allocate_pinned(shape, first)
            output.zero_()
            for column, value in enumerate(values):
                output[: value.size(0), column, ...].copy_(value)
            return output
        output, slot, required = self._allocate(shape, first)
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

    def _allocate(self, shape: tuple[int, ...], template: Any) -> tuple[Any, Any, int]:
        import math
        import torch

        elements = math.prod(shape)
        required = elements * template.element_size()
        reserved = max(required, template.element_size())
        with self._lock:
            if self._arena is None:
                initial = _size_class(reserved, template.element_size())
                self._arena = _hyperloader._NativeArena(
                    initial,
                    max(2, self.worker_count),
                    self.growth,
                )
            slot = self._arena.reserve(reserved)
            self._capacity_bytes = max(self._capacity_bytes, slot.capacity)
        output = torch.frombuffer(
            slot,
            dtype=template.dtype,
            count=elements,
        ).view(shape)
        return output, slot, required

    def _allocate_pinned(self, shape: tuple[int, ...], template: Any) -> Any:
        pool = self._pinned_pool
        if pool is None:
            raise RuntimeError("pinned final-slot allocation is not active")
        stride = _contiguous_stride(shape)
        with self._lock:
            return pool.acquire(shape, stride, template.dtype)


def _size_class(size: int, element_size: int) -> int:
    minimum = max(size, element_size)
    return 1 << (minimum - 1).bit_length()


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = []
    width = 1
    for size in reversed(shape):
        stride.append(width)
        width *= size
    return tuple(reversed(stride))

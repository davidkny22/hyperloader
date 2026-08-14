"""Calibration selection and ownership for pinned delivery resources."""

from __future__ import annotations

from collections.abc import Iterator
from threading import get_ident
from typing import Any
from weakref import WeakSet

from .iterator import PinnedDeliveryIterator
from .pool import PinnedTensorPool
from .registration import HostRegistration, view_sources


class PinnedDelivery:
    """Own either in-place source registration or a reusable staging pool."""

    def __init__(self, loader: Any) -> None:
        calibration = loader._calibration
        self._pricing = None if calibration is None else calibration.staged_copy_tax
        self._requested_memory = loader.delivery_memory
        self.effective_memory = _effective_memory(loader)
        self._registration: HostRegistration | None = None
        self._pool: PinnedTensorPool | None = None
        self._iterators: WeakSet[PinnedDeliveryIterator] = WeakSet()
        self._consumer_thread_ids: set[int] = set()
        self._staging_thread_ids: set[int] = set()
        self._staging_on_consumer_thread = False
        if self.effective_memory != "pinned":
            return
        sources = view_sources(loader)
        if sources:
            registration = HostRegistration(sources)
            if registration.activate():
                self._registration = registration
                return
        direct = getattr(loader._execution_dataset, "enable_pinned_delivery", None)
        if direct is not None and bool(direct()):
            return
        if self._requested_memory == "auto" and not self._staging_is_profitable():
            self.effective_memory = "host"
            return
        self._pool = PinnedTensorPool()

    @property
    def stages(self) -> bool:
        """Return whether registration refusal selected the pinned pool."""
        return self._pool is not None

    @property
    def reports_selection(self) -> bool:
        """Return whether delivery made a calibrated or pinned-memory choice."""
        return self.effective_memory == "pinned" or self._pricing is not None

    def stage(self, value: Any) -> Any:
        """Return registered views unchanged or copy once into the pinned pool."""
        thread_id = get_ident()
        self._staging_thread_ids.add(thread_id)
        if thread_id in self._consumer_thread_ids:
            self._staging_on_consumer_thread = True
        return value if self._pool is None else self._pool.stage(value)

    def bind_consumer_thread(self, thread_id: int) -> None:
        """Record a thread that consumes staged batches."""
        self._consumer_thread_ids.add(thread_id)
        if thread_id in self._staging_thread_ids:
            self._staging_on_consumer_thread = True

    def attach(self, iterator: Iterator[Any]) -> Iterator[Any]:
        """Own one one-ahead staging wrapper when a copy is selected."""
        if not self.stages:
            return iterator
        wrapped = PinnedDeliveryIterator(self, iterator)
        self._iterators.add(wrapped)
        return wrapped

    def report(self) -> dict[str, object]:
        """Return delivery-memory selection and exact registration or copy bytes."""
        return {
            "delivery_memory": self.effective_memory,
            "pinned_registered_bytes": (
                0 if self._registration is None else self._registration.registered_bytes
            ),
            "pinned_staged_bytes": 0 if self._pool is None else self._pool.copied_bytes,
            "staging_copy_nanoseconds": (
                None
                if self._pricing is None
                else self._pricing.staging_copy_nanoseconds
            ),
            "staging_transfer_benefit_nanoseconds": (
                None
                if self._pricing is None
                else self._pricing.transfer_benefit_nanoseconds
            ),
            "staging_profitable": self._staging_is_profitable(),
            "staging_prefetch_depth": 1 if self.stages else 0,
            "staging_thread_count": len(self._staging_thread_ids),
            "staging_on_consumer_thread": self._staging_on_consumer_thread,
        }

    def compose_memory_report(self, memory: dict[str, object]) -> None:
        """Add delivery-stage traffic to an execution-owned memory report."""
        report = self.report()
        memory.update(report)
        staged = int(report["pinned_staged_bytes"])
        if staged == 0 or "actual_bytes" not in memory:
            return
        actual = int(memory["actual_bytes"]) + staged
        overhead = int(memory.get("bytes_beyond_irreducible", 0)) + staged
        samples = int(memory.get("produced_samples", 0))
        memory["actual_bytes"] = actual
        memory["bytes_beyond_irreducible"] = overhead
        memory["actual_bytes_per_sample"] = actual / samples if samples else 0.0
        memory["bytes_beyond_irreducible_per_sample"] = (
            overhead / samples if samples else 0.0
        )

    def close(self) -> None:
        """Release registration and pool ownership."""
        for iterator in tuple(self._iterators):
            iterator.close()
        self._iterators.clear()
        if self._registration is not None:
            self._registration.close()
            self._registration = None
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def _staging_is_profitable(self) -> bool:
        return bool(self._pricing is not None and self._pricing.staging_is_profitable)


def configure_pinned_delivery(loader: Any) -> PinnedDelivery:
    """Resolve and retain one delivery-memory owner per loader."""
    if loader._pinned_delivery is None:
        loader._pinned_delivery = PinnedDelivery(loader)
        loader.delivery_memory = loader._pinned_delivery.effective_memory
    return loader._pinned_delivery


def attach_pinned_delivery(
    delivery: PinnedDelivery, iterator: Iterator[Any]
) -> Iterator[Any]:
    """Wrap only the staged fallback path."""
    return delivery.attach(iterator)


def _effective_memory(loader: Any) -> str:
    requested = loader.delivery_memory
    if requested != "auto":
        return requested
    calibration = loader._calibration
    tax = None if calibration is None else calibration.staged_copy_tax
    if tax is None or tax.transfer_benefit_nanoseconds <= 0:
        return "host"
    import torch

    return "pinned" if torch.cuda.is_available() else "host"

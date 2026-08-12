"""Iterator lifecycle for torch-compatible operating-system worker lanes."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .multi_state import CompatMultiCheckpoint
from .protocol import TaggedBatch
from .rng import capture_torch_source


class CompatMultiIterator(Iterator[Any]):
    """Track strict delivery around torch's real worker architecture."""

    def __init__(
        self,
        loader: Any,
        iterator: Iterator[Any],
        iterator_generator: bytes,
        *,
        delivered: int = 0,
        dummy_batches: int = 0,
        tagged: bool = True,
        reused_base_seed: bool = False,
    ) -> None:
        self._loader = loader
        self._iterator = iterator
        self._iterator_generator = iterator_generator
        self._delivered = delivered
        self._dummy_batches = dummy_batches
        self._tagged = tagged
        self._reused_base_seed = reused_base_seed
        self._lookahead: list[TaggedBatch] = []
        self._complete = False
        self._valid = True

    def __iter__(self) -> CompatMultiIterator:
        return self

    def __next__(self) -> Any:
        if not self._valid:
            raise RuntimeError("compat iterator is no longer active")
        if not self._tagged:
            try:
                value = next(self._iterator)
            except StopIteration:
                self._complete = True
                raise
            self._delivered += 1
            return value
        batch = self._next_tagged()
        self._delivered += 1
        return batch.value

    def _next_tagged(self) -> TaggedBatch:
        if self._lookahead:
            return self._lookahead.pop(0)
        return self._next_from_iterator()

    def _next_from_iterator(self) -> TaggedBatch:
        """Read one real batch without consulting owner-side lookahead."""
        while True:
            try:
                batch = next(self._iterator)
            except StopIteration:
                self._complete = True
                raise
            if not isinstance(batch, TaggedBatch):
                raise TypeError("compat worker returned an untagged batch")
            if batch.dummy:
                if self._dummy_batches == 0:
                    raise RuntimeError("compat worker returned an unexpected dummy batch")
                self._dummy_batches -= 1
                continue
            return batch

    def prime_resume(self) -> None:
        """Force lazy sampler replay before restoring the owner's generator."""
        try:
            self._lookahead.append(self._next_tagged())
        except StopIteration:
            pass

    def capture_checkpoint(self) -> CompatMultiCheckpoint:
        """Capture the first undelivered restore point for every active lane."""
        if not self._tagged:
            raise RuntimeError("compat snapshot capture requires tagged worker lanes")
        lane_states = {batch.worker: batch.state for batch in self._lookahead}
        lane_seeds = {batch.worker: batch.seed for batch in self._lookahead}
        while len(lane_states) < self._loader.num_workers and not self._complete:
            try:
                batch = self._next_from_iterator()
            except StopIteration:
                break
            self._lookahead.append(batch)
            lane_states.setdefault(batch.worker, batch.state)
            lane_seeds.setdefault(batch.worker, batch.seed)
        return CompatMultiCheckpoint(
            delivered_batches=self._delivered,
            worker_count=self._loader.num_workers,
            assignment_phase=self._delivered % self._loader.num_workers,
            reused_base_seed=self._reused_base_seed,
            iterator_generator=self._iterator_generator,
            current_generator=capture_torch_source(
                self._loader._compat_generator
            ),
            lane_states=lane_states,
            lane_seeds=lane_seeds,
            fingerprint=self._loader._fingerprint,
        )

    @property
    def complete(self) -> bool:
        """Report whether the wrapped torch iterator is exhausted."""
        return self._complete

    def invalidate(self, *, shutdown: bool = True) -> None:
        """Stop an abandoned worker iterator and prevent later delivery."""
        if not self._valid:
            return
        self._valid = False
        shutdown_workers = getattr(self._iterator, "_shutdown_workers", None)
        if shutdown and shutdown_workers is not None:
            shutdown_workers()

    def _flush_telemetry(self) -> None:
        """Keep the public telemetry snapshot hook available in compat mode."""

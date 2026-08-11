"""Bounded delivered-batch state for completion-order execution."""

from __future__ import annotations


class DeliveredBatchState:
    """Track a contiguous prefix and delivered batches beyond its first gap."""

    def __init__(self, base: int = 0, ahead: set[int] | None = None) -> None:
        self.base = base
        self._ahead = set() if ahead is None else set(ahead)

    def mark(self, ordinal: int) -> None:
        """Record one newly delivered batch ordinal."""
        if ordinal < self.base or ordinal in self._ahead:
            raise RuntimeError("completion-order delivery repeated a batch")
        if ordinal == self.base:
            self.base += 1
            while self.base in self._ahead:
                self._ahead.remove(self.base)
                self.base += 1
            return
        self._ahead.add(ordinal)

    def bitmap(self) -> bytes:
        """Encode delivered ordinals relative to the contiguous prefix."""
        if not self._ahead:
            return b""
        highest = max(self._ahead) - self.base
        bitmap = bytearray(highest // 8 + 1)
        for ordinal in self._ahead:
            offset = ordinal - self.base
            bitmap[offset // 8] |= 1 << (offset % 8)
        return bytes(bitmap)

    @property
    def ahead(self) -> frozenset[int]:
        """Return delivered ordinals beyond the contiguous prefix."""
        return frozenset(self._ahead)


def decode_delivered_bitmap(base: int, bitmap: bytes, total_batches: int) -> set[int]:
    """Decode and validate delivered ordinals relative to a resume gap."""
    delivered = {
        base + byte_index * 8 + bit
        for byte_index, byte in enumerate(bitmap)
        for bit in range(8)
        if byte & (1 << bit)
    }
    if base > total_batches:
        raise ValueError("loader state cursor exceeds the delivery batch count")
    if base in delivered:
        raise ValueError("delivered_bitmap bit zero must name the first open gap")
    if any(ordinal >= total_batches for ordinal in delivered):
        raise ValueError("delivered_bitmap exceeds the delivery batch count")
    return delivered

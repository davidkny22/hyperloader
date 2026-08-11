"""Bounded delivered-batch state for completion-order execution."""

from __future__ import annotations


class DeliveredBatchState:
    """Track a contiguous prefix and delivered batches beyond its first gap."""

    def __init__(self, base: int = 0) -> None:
        self.base = base
        self._ahead: set[int] = set()

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

"""Map-style epoch transition state."""

from __future__ import annotations


class EpochState:
    """Coordinate iteration starts, deliveries, completion, and explicit overrides."""

    def __init__(self) -> None:
        self._current = 0
        self._active_epoch: int | None = None
        self._active_delivered = False
        self._pending_override: str | None = None

    @property
    def current(self) -> int:
        """Return the epoch assigned to the next iterator."""
        return self._current

    def begin_iteration(self) -> bool:
        """Begin an iterator and report whether abandonment advanced the epoch."""
        advanced = False
        if self._pending_override is None and self._active_delivered:
            self._current += 1
            advanced = True
        self._pending_override = None
        self._active_epoch = self._current
        self._active_delivered = False
        return advanced

    def mark_delivered(self, epoch: int) -> None:
        """Record a successful delivery from the active iterator."""
        if self._active_epoch == epoch:
            self._active_delivered = True

    def complete(self, epoch: int) -> None:
        """Advance a normally exhausted active iterator exactly once."""
        if self._active_epoch != epoch:
            return
        if self._pending_override is None and self._current == epoch:
            self._current += 1
        self._active_epoch = None
        self._active_delivered = False

    def set_epoch(self, epoch: int) -> None:
        """Set an explicit replay or forward epoch for the next iterator."""
        self._validate(epoch)
        self._current = epoch
        self._pending_override = "explicit"

    def restore(self, epoch: int) -> None:
        """Set a restored epoch whose first iterator must not auto-advance."""
        self._validate(epoch)
        self._current = epoch
        self._pending_override = "resume"

    @staticmethod
    def _validate(epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise TypeError("epoch must be an integer")
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")

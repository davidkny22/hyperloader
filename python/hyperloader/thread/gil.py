"""Free-threaded runtime restoration detection."""

from __future__ import annotations

import sys
import threading
from typing import Any


def gil_enabled() -> bool | None:
    """Return the runtime GIL state when CPython exposes the query."""
    query = getattr(sys, "_is_gil_enabled", None)
    return None if query is None else bool(query())


class GilRestorationDetector:
    """Report one process-wide GIL restoration after a free-threaded start."""

    def __init__(self, recorder: Any | None) -> None:
        self._recorder = recorder
        self._started_free_threaded = gil_enabled() is False
        self._reported = False
        self._lock = threading.Lock()

    def observe(self) -> None:
        """Record the first false-to-true runtime transition."""
        if (
            self._recorder is None
            or not self._started_free_threaded
            or self._reported
            or gil_enabled() is not True
        ):
            return
        with self._lock:
            if self._reported:
                return
            self._reported = True
            self._recorder.record_gil_restore()

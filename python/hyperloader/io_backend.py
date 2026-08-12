"""Plan-time routing for the installed file-input backend."""

from __future__ import annotations

from . import _hyperloader


def select_io_backend(preference: str) -> str:
    """Resolve one configured platform backend through the installed engine."""
    return str(_hyperloader._io_backend_kind(preference))

"""Python ownership of the optional native telemetry recorder."""

from __future__ import annotations

from typing import Any

from hyperloader import _hyperloader


def build_telemetry(enabled: bool) -> Any | None:
    """Allocate the native recorder only when instrumentation is enabled."""
    return _hyperloader._Telemetry() if enabled else None


def instrument_registry() -> tuple[dict[str, object], ...]:
    """Return the versioned stable instrument registry."""
    return tuple(dict(entry) for entry in _hyperloader._Telemetry.registry())


def telemetry_snapshot(
    recorder: Any | None,
    controller: dict[str, int | float | str | bool | None] | None,
) -> dict[str, object]:
    """Return a public snapshot while preserving the disabled fast path."""
    snapshot = (
        {
            "current": None,
            "enabled": False,
            "last_epoch": None,
            "registry": instrument_registry(),
            "startup_ns": 0,
        }
        if recorder is None
        else dict(recorder.snapshot())
    )
    snapshot["controller"] = None if controller is None else dict(controller)
    return snapshot

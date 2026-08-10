"""Native telemetry construction and stable instrument metadata."""

from .runtime import build_telemetry, instrument_registry, telemetry_snapshot

__all__ = ["build_telemetry", "instrument_registry", "telemetry_snapshot"]

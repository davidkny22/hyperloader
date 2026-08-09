"""Native instrumentation configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TelemetryConfig:
    """Configure native instruments and benchmark-mode discipline."""

    enabled: bool = True
    benchmark_mode: bool = False

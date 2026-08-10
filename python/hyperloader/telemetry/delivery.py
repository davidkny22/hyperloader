"""Execution-path hooks for the optional native telemetry recorder."""

from __future__ import annotations

import time
from typing import Any


class DeliveryTelemetry:
    """Translate iterator events into native recorder operations."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self._recorder = loader._telemetry
        self._last_delivery_ns = time.perf_counter_ns()

    def record_delivery(self, samples: int, bytes: int, started_ns: int) -> None:
        """Record one successful delivery and its startup boundary."""
        delivered_ns = time.perf_counter_ns()
        self._recorder.record_startup(
            delivered_ns - self._loader._construction_started_ns
        )
        self._recorder.record_delivery(
            samples,
            bytes,
            delivered_ns - started_ns,
            delivered_ns - self._last_delivery_ns,
        )
        self._last_delivery_ns = delivered_ns

    def record_stall(self) -> None:
        """Record one event-driven wait for delivery work."""
        self._recorder.record_stall()

    def record_controller(self, decision: Any) -> None:
        """Record one low-cadence controller decision."""
        self._recorder.record_controller(
            decision.previous_width,
            decision.width,
            decision.reason,
            decision.starvation,
            decision.score[1],
            decision.binding,
        )

    def finish_epoch(self, epoch: int) -> None:
        """Seal the current counters as one epoch summary."""
        self._recorder.finish_epoch(epoch)


def build_delivery_telemetry(loader: Any) -> DeliveryTelemetry | None:
    """Keep the disabled iterator path free of telemetry helper allocation."""
    return None if loader._telemetry is None else DeliveryTelemetry(loader)

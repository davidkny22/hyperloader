"""Execution-path hooks for the optional native telemetry recorder."""

from __future__ import annotations

import time
from typing import Any

DELIVERY_GROUP = 256


class DeliveryTelemetry:
    """Translate iterator events into native recorder operations."""

    def __init__(self, loader: Any) -> None:
        self._recorder = loader._telemetry
        self._pending_samples = 0
        self._pending_batches = 0
        self._pending_bytes = 0
        self._pending_latency_ns = 0
        self._startup_pending = True

    def start_delivery(self) -> int:
        """Start a latency sample only at an event-group boundary."""
        return time.perf_counter_ns() if self._pending_batches == 0 else 0

    def record_delivery(self, samples: int, bytes: int, started_ns: int) -> None:
        """Record one successful delivery and its startup boundary."""
        if self._pending_batches == 0:
            self._pending_latency_ns = time.perf_counter_ns() - started_ns
        if self._startup_pending:
            self._recorder.record_startup_now()
            self._startup_pending = False
        self._pending_samples += samples
        self._pending_batches += 1
        self._pending_bytes += bytes
        if self._pending_batches == DELIVERY_GROUP:
            self.flush()

    def flush(self) -> None:
        """Publish a pending event group before a snapshot or epoch boundary."""
        if self._pending_batches == 0:
            return
        self._recorder.record_deliveries(
            self._pending_samples,
            self._pending_batches,
            self._pending_bytes,
            self._pending_latency_ns,
        )
        self._pending_samples = 0
        self._pending_batches = 0
        self._pending_bytes = 0
        self._pending_latency_ns = 0

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
        self.flush()
        self._recorder.finish_epoch(epoch)


def build_delivery_telemetry(loader: Any) -> DeliveryTelemetry | None:
    """Keep the disabled iterator path free of telemetry helper allocation."""
    return None if loader._telemetry is None else DeliveryTelemetry(loader)

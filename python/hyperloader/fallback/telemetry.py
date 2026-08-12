"""Pure-Python telemetry aggregation with the stable public schema."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

_INSTRUMENTS = (
    ("bytes_per_sample", "bytes", "Measured sample payload size."),
    ("nontensor_fraction", "ratio", "Fraction of non-tensor leaves."),
    ("t_shim", "nanoseconds", "Per-sample Python shim time."),
    ("t_seed", "nanoseconds", "Per-sample RNG install time."),
    ("hold", "events", "Consumer-held arena recycle events."),
    ("growth_events", "events", "Arena or frontier growth events."),
    ("view_export_copy", "bytes", "Bytes copied while exporting views."),
    ("overflow_events", "events", "Overflow-slab sample events."),
    (
        "staged_copy_transients",
        "bytes",
        "Transient bytes copied before stable layout selection.",
    ),
    ("stall_events", "events", "Delivery waits for unavailable work."),
    ("hung_position", "position", "Position named by a liveness timeout."),
    (
        "ceiling_binds",
        "events",
        "Controller decisions bound by user ceilings.",
    ),
    (
        "gil_restore_events",
        "events",
        "Thread-tier GIL restoration events.",
    ),
    ("delivery_rate", "samples_per_second", "Delivered sample rate."),
    (
        "delivery_latency",
        "nanoseconds",
        "Event-sampled successful next-batch latency.",
    ),
    ("startup", "nanoseconds", "Construction-to-first-delivery latency."),
    ("controller_decisions", "events", "Inspectable controller decisions."),
    (
        "machine_keeping_duty",
        "ratio",
        "Current native consumer keep-warm duty.",
    ),
)


class Telemetry:
    """Aggregate bounded counters and delivery latency buckets."""

    def __init__(self) -> None:
        now = time.perf_counter_ns()
        self._constructed_at = now
        self._last_delivery_at = now
        self._startup_ns = 0
        self._last_epoch: dict[str, Any] | None = None
        self._reset_current()

    @staticmethod
    def registry() -> list[dict[str, object]]:
        """Return stable instrument metadata."""
        return [
            {
                "description": description,
                "id": identifier,
                "unit": unit,
                "version": 1,
            }
            for identifier, unit, description in _INSTRUMENTS
        ]

    def record_startup(self, nanoseconds: int) -> None:
        """Retain the maximum observed startup duration."""
        self._startup_ns = max(self._startup_ns, nanoseconds)

    def record_startup_now(self) -> None:
        """Measure startup from recorder construction."""
        now = time.perf_counter_ns()
        self.record_startup(now - self._constructed_at)
        self._last_delivery_at = now

    def record_delivery(self, samples: int, bytes: int, latency_ns: int) -> None:
        """Record one delivered batch."""
        self.record_deliveries(samples, 1, bytes, latency_ns)

    def record_deliveries(
        self, samples: int, batches: int, bytes: int, latency_ns: int
    ) -> None:
        """Record a delivery group and its latency."""
        now = time.perf_counter_ns()
        self.record_startup(now - self._constructed_at)
        self._record_counts(samples, batches, bytes, now)
        self._latencies.append(_latency_bucket(latency_ns))

    def record_counts(self, samples: int, batches: int, bytes: int) -> None:
        """Record a delivery group without a latency sample."""
        now = time.perf_counter_ns()
        self.record_startup(now - self._constructed_at)
        self._record_counts(samples, batches, bytes, now)

    def record_stall(self) -> None:
        """Record one consumer-visible stall."""
        self._stall_events += 1

    def record_gil_restore(self) -> None:
        """Record one detected GIL restoration."""
        self._gil_restore_events += 1

    def record_controller(
        self,
        previous_width: int,
        width: int,
        reason: str,
        starvation: bool,
        resource_loss: float,
        binding: str | None = None,
    ) -> None:
        """Record one controller decision."""
        self._decisions.append(
            {
                "binding": binding,
                "previous_width": previous_width,
                "reason": reason,
                "resource_loss": resource_loss,
                "starvation": starvation,
                "width": width,
            }
        )

    def finish_epoch(self, epoch: int) -> None:
        """Seal the current counters and reset the active epoch."""
        self._last_epoch = self._summary(epoch)
        self._reset_current()

    def snapshot(self) -> dict[str, object]:
        """Return a detached public telemetry snapshot."""
        return {
            "current": self._summary(0),
            "enabled": True,
            "last_epoch": deepcopy(self._last_epoch),
            "registry": self.registry(),
            "startup_ns": self._startup_ns,
        }

    def _record_counts(self, samples: int, batches: int, bytes: int, now: int) -> None:
        self._delivered_samples += samples
        self._delivered_batches += batches
        self._delivered_bytes += bytes
        self._delivery_interval_ns += now - self._last_delivery_at
        self._last_delivery_at = now

    def _summary(self, epoch: int) -> dict[str, object]:
        latencies = sorted(self._latencies)

        def quantile(fraction: float) -> int:
            if not latencies:
                return 0
            index = min(len(latencies) - 1, int(fraction * len(latencies)))
            return latencies[index]

        return {
            "ceiling_binds": sum(
                decision["binding"] is not None for decision in self._decisions
            ),
            "controller_decisions": deepcopy(self._decisions),
            "delivered_batches": self._delivered_batches,
            "delivered_bytes": self._delivered_bytes,
            "delivered_samples": self._delivered_samples,
            "delivery_latency_ns": {
                "p50": quantile(0.50),
                "p95": quantile(0.95),
                "p99": quantile(0.99),
            },
            "delivery_rate": (
                0.0
                if self._delivery_interval_ns == 0
                else self._delivered_samples
                * 1_000_000_000.0
                / self._delivery_interval_ns
            ),
            "epoch": epoch,
            "gil_restore_events": self._gil_restore_events,
            "stall_events": self._stall_events,
        }

    def _reset_current(self) -> None:
        self._delivered_samples = 0
        self._delivered_batches = 0
        self._delivered_bytes = 0
        self._delivery_interval_ns = 0
        self._latencies: list[int] = []
        self._stall_events = 0
        self._gil_restore_events = 0
        self._decisions: list[dict[str, object]] = []


def _latency_bucket(value: int) -> int:
    if value <= 0:
        return 0
    return (1 << value.bit_length()) - 1

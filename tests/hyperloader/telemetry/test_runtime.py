"""Native telemetry registry and snapshot checks."""

from __future__ import annotations

import unittest

from hyperloader import _hyperloader
from hyperloader.telemetry import instrument_registry


class TelemetryRuntimeTest(unittest.TestCase):
    """Exercise stable metadata and bounded native aggregation."""

    def test_registry_is_unique_versioned_and_total_for_named_instruments(self) -> None:
        registry = instrument_registry()
        identifiers = [entry["id"] for entry in registry]

        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual({entry["version"] for entry in registry}, {1})
        self.assertTrue(
            {
                "bytes_per_sample",
                "ceiling_binds",
                "controller_decisions",
                "delivery_latency",
                "delivery_rate",
                "gil_restore_events",
                "growth_events",
                "hold",
                "hung_position",
                "nontensor_fraction",
                "overflow_events",
                "staged_copy_transients",
                "stall_events",
                "startup",
                "t_seed",
                "t_shim",
                "view_export_copy",
            }.issubset(identifiers)
        )

    def test_native_snapshot_seals_epoch_and_resets_current_counters(self) -> None:
        recorder = _hyperloader._Telemetry()
        recorder.record_startup(90)
        for latency in (10, 20, 30, 40):
            recorder.record_delivery(2, 16, latency)
        recorder.record_stall()
        recorder.record_controller(2, 1, "bandwidth-ceiling", True, 0.02, "bandwidth")

        current = recorder.snapshot()["current"]
        self.assertEqual(current["delivered_samples"], 8)
        self.assertEqual(current["delivery_latency_ns"], {"p50": 31, "p95": 63, "p99": 63})
        self.assertEqual(current["ceiling_binds"], 1)

        recorder.finish_epoch(7)
        snapshot = recorder.snapshot()
        self.assertEqual(snapshot["current"]["delivered_samples"], 0)
        self.assertEqual(snapshot["last_epoch"]["epoch"], 7)


if __name__ == "__main__":
    unittest.main()

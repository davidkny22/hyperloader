"""Acceptance checks for Spark overhead evidence."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from benches.overhead_results import clock_samples_valid, summarize_splits
from benches.benchmark_protocol import EnvironmentMetadata
from benches.overhead_feeders import resident_batch_count

BENCHES = Path(__file__).parents[3] / "benches"


class OverheadResultsTest(unittest.TestCase):
    """Reject unpinned cells and aggregate exact copy-rate records."""

    def test_clock_check_requires_a_positive_loaded_sample(self) -> None:
        self.assertTrue(
            clock_samples_valid(
                [
                    {"clock_mhz": 208, "utilization_percent": 0},
                    {"clock_mhz": 2255, "utilization_percent": 98},
                ]
            )
        )
        self.assertFalse(
            clock_samples_valid([{"clock_mhz": 0, "utilization_percent": 99}])
        )
        self.assertFalse(
            clock_samples_valid([{"clock_mhz": 208, "utilization_percent": 0}])
        )

    def test_split_summary_averages_equal_duration_cells(self) -> None:
        cells = [
            {
                "raw": {
                    "byte_split": {
                        "model_input_gbps": value,
                        "irreducible_host_gbps": value * 2,
                        "explicit_overhead_gbps": value * 3,
                        "explicit_total_host_gbps": value * 5,
                    }
                }
            }
            for value in (1.0, 3.0)
        ]
        summary = summarize_splits(cells)
        self.assertEqual(summary["model_input_gbps"], 2.0)
        self.assertEqual(summary["irreducible_host_gbps"], 4.0)
        self.assertEqual(summary["explicit_overhead_gbps"], 6.0)
        self.assertIn("Python serialization", str(summary["overhead_scope"]))
        self.assertIn("fixed-text", str(summary["stage_plan_pin"]))

    def test_resident_ring_covers_eight_times_total_llc(self) -> None:
        llc_bytes = 24 * 1024 * 1024
        batch_bytes = 64 * 512 * 8
        batches = resident_batch_count(llc_bytes, batch_bytes)
        self.assertEqual(batches, 768)
        self.assertGreaterEqual(batches * batch_bytes, 8 * llc_bytes)

    def test_rejected_cell_is_persisted_before_clock_acceptance(self) -> None:
        import sys

        sys.path.insert(0, str(BENCHES))
        try:
            import overhead_campaign

            environment = EnvironmentMetadata(
                captured_at="2026-08-09T00:00:00+00:00",
                machine="spark",
                operating_system="Linux",
                kernel="6.11",
                architecture="aarch64",
                python="3.12.3",
                commit="abcdef0",
                cpu_governor="performance",
                gpu_clock="locked-2400MHz",
                cache_regime="warm",
                benchmark_mode=True,
                concurrent_load=False,
            )
            rejected = {
                "raw": {
                    "clock_samples": [
                        {"clock_mhz": 2392, "utilization_percent": 0}
                    ]
                }
            }
            with TemporaryDirectory() as directory, patch.object(
                overhead_campaign, "run_cell", return_value=rejected
            ):
                output = Path(directory)
                with self.assertRaisesRegex(RuntimeError, "clock samples"):
                    overhead_campaign._run_regime(
                        regime="compute",
                        output=output,
                        environment=environment,
                        threshold_percent=2.0,
                        half_seconds=1.0,
                        llc_bytes=24 * 1024 * 1024,
                        smoke=True,
                    )
                self.assertIn("2392", (output / "compute-cells.jsonl").read_text())
        finally:
            sys.path.remove(str(BENCHES))


if __name__ == "__main__":
    unittest.main()

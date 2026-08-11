"""GPU timeline and clock-residency diagnostic helpers."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

BENCHES = Path(__file__).parents[3] / "benches"
sys.path.insert(0, str(BENCHES))
clock_residency = importlib.import_module("dominance_clock_residency")
segments = importlib.import_module("dominance_gpu_segments")
workloads = importlib.import_module("overhead_workload")


class DominanceGpuTimelineTest(unittest.TestCase):
    def test_segment_summary_preserves_each_timing_family(self) -> None:
        observations = [
            {
                key: float(index + offset)
                for index, key in enumerate(segments.SEGMENT_KEYS)
            }
            for offset in (1, 2, 3)
        ]

        summary = segments.summarize_segments(observations)

        self.assertEqual(summary["iterations"], 3)
        self.assertEqual(summary["cuda_copy_ms"]["mean"], 2.0)
        self.assertEqual(summary["cuda_copy_ms"]["p50"], 2.0)
        self.assertGreater(summary["gpu_operation_iterations_per_second"], 0.0)

    def test_clock_samples_are_assigned_to_ordered_halves(self) -> None:
        samples = [
            {"elapsed_seconds": 1.0, "clock_mhz": 2380, "utilization_percent": 50},
            {"elapsed_seconds": 44.0, "clock_mhz": 2360, "utilization_percent": 55},
            {"elapsed_seconds": 46.0, "clock_mhz": 2340, "utilization_percent": 75},
            {"elapsed_seconds": 89.0, "clock_mhz": 2330, "utilization_percent": 77},
        ]

        report = segments.split_clock_samples(samples, ("hyperloader", "torch"), 45.0)

        self.assertEqual(report["hyperloader"]["samples"], 2)
        self.assertEqual(report["torch"]["samples"], 2)
        self.assertEqual(report["torch"]["residency"]["below_2350_percent"], 100.0)

    def test_existing_cell_extraction_respects_alternating_order(self) -> None:
        rows = []
        for ordinal, order in enumerate(
            (("hyperloader", "torch"), ("torch", "hyperloader"))
        ):
            rows.append(
                {
                    "ordinal": ordinal,
                    "first": {"system": order[0], "duration_seconds": 45.0},
                    "second": {"system": order[1], "duration_seconds": 45.0},
                    "raw": {
                        "clock_samples": [
                            {
                                "elapsed_seconds": 1.0,
                                "clock_mhz": 2380,
                                "utilization_percent": 50,
                            },
                            {
                                "elapsed_seconds": 46.0,
                                "clock_mhz": 2340,
                                "utilization_percent": 75,
                            },
                        ]
                    },
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cells.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            report = clock_residency.extract_residency(path)

        self.assertEqual(report["aggregate"]["hyperloader"]["samples"], 2)
        self.assertEqual(report["aggregate"]["torch"]["samples"], 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA timing events are required")
    def test_gpu_workload_reports_cuda_and_host_segments(self) -> None:
        workload = workloads.GpuWorkload("compute")
        batch = torch.arange(64 * 512, dtype=torch.int64).reshape(64, 512)

        report = workload.run_timed(batch)

        self.assertGreaterEqual(report["cuda_copy_ms"], 0.0)
        self.assertGreater(report["cuda_kernel_ms"], 0.0)
        self.assertGreater(report["host_sync_ms"], 0.0)
        self.assertGreater(report["host_total_ms"], 0.0)
        self.assertAlmostEqual(
            report["cuda_total_ms"],
            report["cuda_copy_ms"] + report["cuda_kernel_ms"],
            places=3,
        )


if __name__ == "__main__":
    unittest.main()

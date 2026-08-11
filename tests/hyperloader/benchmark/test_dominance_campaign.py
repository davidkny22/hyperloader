"""Selected comparison summaries for the dominance campaign."""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

BENCHES = Path(__file__).parents[3] / "benches"
sys.path.insert(0, str(BENCHES))

summarize_results = importlib.import_module("dominance_campaign").summarize_results
HyperloaderFeeder = importlib.import_module("dominance_feeders").HyperloaderFeeder
cpu_idle = importlib.import_module("dominance_cpu_idle")


class DominanceCampaignTest(unittest.TestCase):
    """Verify selected comparisons retain an explicit decision criterion."""

    def test_selected_campaign_requires_every_selected_comparison(self) -> None:
        results = {
            "fixed-text": {
                "torch": {"status": "win"},
                "spdl": {"status": "tie"},
            }
        }
        summary = summarize_results(
            results,
            workloads=("fixed-text",),
            references=("torch", "spdl"),
            smoke=False,
        )
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["required_workloads"], 1)
        results["fixed-text"]["torch"]["status"] = "loss"
        self.assertEqual(
            summarize_results(
                results,
                workloads=("fixed-text",),
                references=("torch", "spdl"),
                smoke=False,
            )["status"],
            "fail",
        )

    def test_complete_matrix_keeps_the_five_workload_threshold(self) -> None:
        names = (
            "images-light",
            "images-heavy",
            "fixed-text",
            "varlen-text",
            "arrow-tabular",
            "numpy-array",
        )
        results = {
            name: {
                "torch": {"status": "win" if index < 5 else "loss"},
                "spdl": {"status": "tie" if index < 5 else "loss"},
            }
            for index, name in enumerate(names)
        }
        summary = summarize_results(
            results,
            workloads=names,
            references=("torch", "spdl"),
            smoke=False,
        )
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["required_workloads"], 5)

    def test_hyperloader_report_uses_public_delivery_instruments(self) -> None:
        feeder = object.__new__(HyperloaderFeeder)
        feeder.batches = 17
        feeder._loader = SimpleNamespace(
            delivery_memory="pinned",
            stats=lambda: {
                "current": {"machine_keeping_duty": 0.03},
                "memory": {
                    "pinned_registered_bytes": 8_388_608,
                    "pinned_staged_bytes": 0,
                },
            },
        )
        report = feeder.report()
        self.assertEqual(report["delivery_memory"], "pinned")
        self.assertEqual(report["stats"]["current"]["machine_keeping_duty"], 0.03)
        self.assertEqual(
            report["stats"]["memory"]["pinned_registered_bytes"], 8_388_608
        )

    def test_cpu_idle_sampler_reports_each_uninterrupted_half(self) -> None:
        snapshots = [
            _cpuidle_snapshot(1_000_000_000, 10),
            _cpuidle_snapshot(2_000_000_000, 13),
            _cpuidle_snapshot(3_000_000_000, 18),
        ]
        with (
            patch.object(cpu_idle, "snapshot_cpuidle", side_effect=snapshots),
            patch.object(
                cpu_idle,
                "snapshot_named_thread_cpu_seconds",
                side_effect=(0.0, 0.05, 0.15),
            ),
        ):
            sampler = cpu_idle.HalfBoundaryCpuIdleSampler()
            sampler.start(0.0)
            report = sampler.stop()

        first = report["first"]["rows"][0]
        second = report["second"]["rows"][0]
        self.assertEqual(first["usage_delta"], 3)
        self.assertEqual(second["usage_delta"], 5)
        self.assertEqual(report["first"]["duration_seconds"], 1.0)
        self.assertEqual(report["second"]["duration_seconds"], 1.0)
        self.assertEqual(report["first"]["machine_keeper_cpu_seconds"], 0.05)
        self.assertAlmostEqual(report["second"]["machine_keeper_cpu_seconds"], 0.1)


def _cpuidle_snapshot(captured_ns: int, usage: int) -> dict[str, object]:
    return {
        "captured_monotonic_ns": captured_ns,
        "cpus": {
            "19": {
                "1": {
                    "name": "LPI-1",
                    "description": "CoreIdle-OFF",
                    "exit_latency_us": 42,
                    "target_residency_us": 1_930,
                    "time_us": usage * 2_000,
                    "usage": usage,
                }
            }
        },
    }


if __name__ == "__main__":
    unittest.main()

"""Selected comparison summaries for the dominance campaign."""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

BENCHES = Path(__file__).parents[3] / "benches"
sys.path.insert(0, str(BENCHES))

summarize_results = importlib.import_module("dominance_campaign").summarize_results
HyperloaderFeeder = importlib.import_module("dominance_feeders").HyperloaderFeeder


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


if __name__ == "__main__":
    unittest.main()

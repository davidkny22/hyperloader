"""Tests for the passive observer overhead verifier."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[3]
PATH = ROOT / "benches" / "observer_overhead_report.py"
SPEC = importlib.util.spec_from_file_location("observer_overhead_report_test", PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {PATH}")
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


def _pair(order: str, penalty: float = 0.0) -> dict[str, object]:
    baseline = {"checksum": 17, "cpu_ns": 90_000_000, "wall_ns": 100_000_000}
    observed = {
        "checksum": 17,
        "cpu_ns": 90_001_000,
        "wall_ns": round(100_000_000 * (1 + penalty)),
    }
    left, right = (
        (observed, baseline) if order == "observer-first" else (baseline, observed)
    )
    return {
        "left_checksum": left["checksum"],
        "left_cpu_ns": left["cpu_ns"],
        "left_wall_ns": left["wall_ns"],
        "order": order,
        "right_checksum": right["checksum"],
        "right_cpu_ns": right["cpu_ns"],
        "right_wall_ns": right["wall_ns"],
    }


def _report(penalty: float = 0.0) -> dict[str, object]:
    pairs = [
        _pair("observer-first" if index % 2 == 0 else "baseline-first", penalty)
        for index in range(20)
    ]
    noise = [
        _pair("null", 0.0005 if index % 2 == 0 else -0.0005) for index in range(20)
    ]
    cells = {"pairs": pairs, "noise_pairs": noise}
    return {
        "active_probe": {
            "requested_batches": 4,
            "consumed_batches": 4,
            "elapsed_ns": 1000,
        },
        "loaders": {
            "hyperloader": cells,
            "torch": {
                "pairs": [dict(pair) for pair in pairs],
                "noise_pairs": [dict(pair) for pair in noise],
            },
        },
        "metadata": {
            "batch_size": 64,
            "batches_per_half": 512,
            "pair_count": 20,
            "platform": "test",
            "public_path_verified": True,
            "python": "3.12.test",
            "torch": "2.test",
        },
    }


class ObserverOverheadReportTest(unittest.TestCase):
    def test_below_noise_passes(self) -> None:
        result = REPORT.evaluate_report(_report())
        self.assertEqual(result["decision"], "PASS")
        self.assertTrue(result["loaders"]["hyperloader"]["below_noise"])

    def test_detectable_penalty_fails(self) -> None:
        self.assertEqual(REPORT.evaluate_report(_report(0.01))["decision"], "FAIL")

    def test_named_assumptions_are_rejected(self) -> None:
        cases = (
            ("pair_count", 19, "pair count"),
            ("public_path_verified", False, "installed artifact"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                report = _report()
                report["metadata"][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    REPORT.validate_report(report)

    def test_delivery_and_probe_assumptions_are_rejected(self) -> None:
        report = _report()
        report["loaders"]["torch"]["pairs"][0]["right_checksum"] = 18
        with self.assertRaisesRegex(ValueError, "identical values"):
            REPORT.validate_report(report)
        report = _report()
        report["active_probe"]["consumed_batches"] = 3
        with self.assertRaisesRegex(ValueError, "probe consumption"):
            REPORT.validate_report(report)

    def test_order_and_duration_are_rejected(self) -> None:
        report = _report()
        report["loaders"]["hyperloader"]["pairs"][1]["order"] = "observer-first"
        with self.assertRaisesRegex(ValueError, "alternate"):
            REPORT.validate_report(report)
        report = _report()
        report["loaders"]["torch"]["noise_pairs"][0]["left_wall_ns"] = 1
        with self.assertRaisesRegex(ValueError, "too short"):
            REPORT.validate_report(report)

    def test_planted_cost_mutation_is_red(self) -> None:
        os.environ["HYPERLOADER_OBSERVER_MUTATION"] = "inflate-cost"
        try:
            self.assertEqual(REPORT.evaluate_report(_report())["decision"], "FAIL")
        finally:
            os.environ.pop("HYPERLOADER_OBSERVER_MUTATION", None)


if __name__ == "__main__":
    unittest.main()

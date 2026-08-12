"""Tests for the fallback no-regression decision surface."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from torch.utils.data import default_collate

ROOT = Path(__file__).parents[3]
PATH = ROOT / "benches" / "fallback_no_regression_report.py"
SPEC = importlib.util.spec_from_file_location("fallback_no_regression_report_test", PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {PATH}")
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)

WORKLOAD_PATH = ROOT / "benches" / "fallback_workloads.py"
WORKLOAD_SPEC = importlib.util.spec_from_file_location(
    "fallback_no_regression_workloads_test", WORKLOAD_PATH
)
if WORKLOAD_SPEC is None or WORKLOAD_SPEC.loader is None:
    raise RuntimeError(f"cannot load {WORKLOAD_PATH}")
WORKLOADS = importlib.util.module_from_spec(WORKLOAD_SPEC)
sys.modules[WORKLOAD_SPEC.name] = WORKLOADS
WORKLOAD_SPEC.loader.exec_module(WORKLOADS)


def _run(elapsed_ns: int, checksum: int = 17) -> dict[str, int]:
    return {"checksum": checksum, "elapsed_ns": elapsed_ns, "samples": 1_024}


def _report(fallback_ns: int = 90, torch_ns: int = 100) -> dict[str, object]:
    workloads = {}
    for name in ("fixed-record", "numpy-array"):
        workloads[name] = [
            {
                "fallback": _run(fallback_ns),
                "order": "fallback-first" if ordinal % 2 == 0 else "torch-first",
                "ordinal": ordinal,
                "torch": _run(torch_ns),
            }
            for ordinal in range(10)
        ]
    return {
        "metadata": {
            "equal_tuning": True,
            "fallback_resolved": True,
            "pair_count": 10,
            "public_path_verified": True,
        },
        "workloads": workloads,
    }


class FallbackNoRegressionReportTest(unittest.TestCase):
    """Require strict non-regression and reject incomparable records."""

    def test_faster_fallback_passes_both_workloads(self) -> None:
        decision = REPORT.evaluate_report(_report(), draws=100)
        self.assertEqual(decision["decision"], "PASS")
        self.assertTrue(
            all(result["lower_percent"] > 0 for result in decision["workloads"].values())
        )

    def test_slower_fallback_and_planted_mutation_fail(self) -> None:
        self.assertEqual(
            REPORT.evaluate_report(_report(110, 100), draws=100)["decision"],
            "FAIL",
        )
        self.assertEqual(
            REPORT.evaluate_report(_report(), mutate=True, draws=100)["decision"],
            "FAIL",
        )

    def test_named_assumptions_are_rejected(self) -> None:
        cases = (
            ("public_path_verified", False, "installed public path"),
            ("fallback_resolved", False, "fallback did not resolve"),
            ("equal_tuning", False, "equal tuning"),
            ("pair_count", 9, "pair count"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                report = _report()
                report["metadata"][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    REPORT.validate_report(report)

    def test_order_counts_and_values_are_enforced(self) -> None:
        report = _report()
        report["workloads"]["fixed-record"][1]["order"] = "fallback-first"
        with self.assertRaisesRegex(ValueError, "pair order"):
            REPORT.validate_report(report)

        report = _report()
        report["workloads"]["numpy-array"][0]["torch"]["checksum"] = 18
        with self.assertRaisesRegex(ValueError, "identical values"):
            REPORT.validate_report(report)

    def test_workload_checksums_follow_default_collation_shapes(self) -> None:
        for workload in WORKLOADS.WORKLOADS:
            with self.subTest(workload=workload.name):
                dataset = workload.dataset(2)
                batch = default_collate([dataset[0], dataset[1]])
                self.assertIsInstance(workload.checksum(batch), int)


if __name__ == "__main__":
    unittest.main()

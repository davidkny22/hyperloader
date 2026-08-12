"""Shim-floor decision verifier tests."""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parents[3]
BENCHES = str(ROOT / "benches")
SPEC = importlib.util.spec_from_file_location(
    "shim_floor_report", ROOT / "benches" / "shim_floor_report.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, BENCHES)
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.remove(BENCHES)


def write_report(
    path: Path,
    core: int,
    *,
    shim_total_ns: float = 1_500.0,
    seed_no_draw_ns: float = 500.0,
    seed_component_ns: float = 1_000.0,
    seed_all_ns: float = 2_000.0,
    numpy_version: str = "numpy-version-from-record",
) -> None:
    """Write one complete deterministic paired-core fixture."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for key, value in {
            "core": core,
            "governor": "performance",
            "python": "runtime-version-from-record",
            "torch": "torch-version-from-record",
            "numpy": numpy_version,
            "gil_disabled_build": "False",
            "gil_enabled": "True",
            "iterations": 100,
            "warmup_iterations": 10,
            "trials": 10,
        }.items():
            writer.writerow(("meta", key, value))
        writer.writerow(
            (
                "kind",
                "metric",
                "trial",
                "iterations",
                "elapsed_ns",
                "ns_per_op",
                "checksum",
            )
        )
        for metric in MODULE.METRICS:
            if metric.endswith("seed_no_draw"):
                value = seed_no_draw_ns
            elif metric.endswith("seed_all"):
                value = seed_all_ns
            elif "_seed_" in metric:
                value = seed_component_ns
            else:
                value = shim_total_ns
            for trial in range(10):
                writer.writerow(
                    ("data", metric, trial, 100, int(value * 100), value, trial + 1)
                )


class ShimFloorReportTest(unittest.TestCase):
    """Verify bounds, aggregate conversion, and invalid report rejection."""

    def test_all_three_endpoints_decide_pass_and_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            performance = root / "performance.csv"
            efficiency = root / "efficiency.csv"
            write_report(performance, 19)
            write_report(
                efficiency,
                0,
                shim_total_ns=2_500.0,
                seed_no_draw_ns=1_000.0,
                seed_all_ns=4_000.0,
            )
            passing = MODULE.evaluate_pair(performance, efficiency, 19, 0)
            self.assertEqual(passing["decision"], "PASS")
            self.assertLessEqual(
                passing["decisions"]["process"]["aggregate_eff_core_equivalents"],
                MODULE.AGGREGATE_LIMIT,
            )
            self.assertEqual(
                set(passing["performance"]["process"]["seed_components"]),
                {"torch", "numpy", "random"},
            )
            self.assertEqual(
                passing["performance"]["process"]["seed_components"]["torch"][
                    "mean_ns"
                ],
                500.0,
            )

            write_report(performance, 19, shim_total_ns=3_000.0)
            self.assertEqual(
                MODULE.evaluate_pair(performance, efficiency, 19, 0)["decision"],
                "FAIL",
            )
            write_report(performance, 19, seed_all_ns=7_000.0)
            self.assertEqual(
                MODULE.evaluate_pair(performance, efficiency, 19, 0)["decision"],
                "FAIL",
            )

    def test_wrong_core_and_incomplete_metrics_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.csv"
            write_report(path, 19)
            with self.assertRaisesRegex(ValueError, "expected core"):
                MODULE.read_report(path, 18)
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text(
                "\n".join(line for line in lines if ",thread_seed_all," not in line)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "complete metric set"):
                MODULE.read_report(path, 19)

    def test_corrupt_rows_and_metadata_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.csv"
            write_report(original, 19)
            lines = original.read_text(encoding="utf-8").splitlines()

            cases = (
                (
                    "unknown",
                    [
                        line.replace("process_seed_no_draw", "foreign_metric", 1)
                        for line in lines
                    ],
                    "unknown shim-floor metric",
                ),
                (
                    "zero",
                    [
                        line.rsplit(",", 1)[0] + ",0"
                        if ",process_seed_no_draw,0," in line
                        else line
                        for line in lines
                    ],
                    "zero checksum",
                ),
                (
                    "governor",
                    [
                        line.replace(
                            "meta,governor,performance", "meta,governor,powersave"
                        )
                        for line in lines
                    ],
                    "governor",
                ),
                (
                    "trials",
                    [line.replace("meta,trials,10", "meta,trials,9") for line in lines],
                    "every required trial",
                ),
            )
            for name, changed, message in cases:
                with self.subTest(name=name):
                    path = root / f"{name}.csv"
                    path.write_text("\n".join(changed) + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        MODULE.read_report(path, 19)

            duplicate = root / "duplicate.csv"
            duplicate.write_text(
                "\n".join(
                    (
                        *lines,
                        next(
                            line for line in lines if ",process_seed_no_draw,0," in line
                        ),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate trial"):
                MODULE.read_report(duplicate, 19)

    def test_tier_vector_and_runtime_pairing_assumptions_are_checked(self) -> None:
        values = {
            "process_seed_no_draw": [1.0],
            "process_seed_torch": [2.0],
            "process_seed_numpy": [2.0],
            "process_seed_random": [2.0],
            "process_seed_all": [2.0, 2.0],
            "process_shim_total": [3.0],
        }
        with self.assertRaisesRegex(ValueError, "differ in length"):
            MODULE.tier_statistics(values, "process")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            performance = root / "performance.csv"
            efficiency = root / "efficiency.csv"
            write_report(performance, 19)
            write_report(
                efficiency,
                0,
                numpy_version="different-numpy-version-from-record",
            )
            with self.assertRaisesRegex(ValueError, "disagree on numpy"):
                MODULE.evaluate_pair(performance, efficiency, 19, 0)

            write_report(performance, 19)
            write_report(efficiency, 0)
            for path in (performance, efficiency):
                path.write_text(
                    path.read_text(encoding="utf-8")
                    .replace(
                        "meta,gil_disabled_build,False", "meta,gil_disabled_build,True"
                    )
                    .replace("meta,gil_enabled,True", "meta,gil_enabled,False"),
                    encoding="utf-8",
                )
            self.assertEqual(
                MODULE.evaluate_pair(performance, efficiency, 19, 0)["runtime"][
                    "gil_enabled"
                ],
                "False",
            )
            performance.write_text(
                performance.read_text(encoding="utf-8").replace(
                    "meta,gil_enabled,False", "meta,gil_enabled,True"
                ),
                encoding="utf-8",
            )
            efficiency.write_text(
                efficiency.read_text(encoding="utf-8").replace(
                    "meta,gil_enabled,False", "meta,gil_enabled,True"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "restored the GIL"):
                MODULE.evaluate_pair(performance, efficiency, 19, 0)

    def test_planted_cost_inflation_turns_the_decision_red(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            performance = root / "performance.csv"
            efficiency = root / "efficiency.csv"
            write_report(performance, 19)
            write_report(efficiency, 0)
            with mock.patch.dict(
                "os.environ", {"HYPERLOADER_SHIM_FLOOR_MUTATION": "inflate-cost"}
            ):
                report = MODULE.evaluate_pair(performance, efficiency, 19, 0)
            self.assertEqual(report["decision"], "FAIL")


if __name__ == "__main__":
    unittest.main()

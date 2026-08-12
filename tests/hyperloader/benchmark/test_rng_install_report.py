"""Verifier tests for RNG install report validation and decisions."""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[3]
SPEC = importlib.util.spec_from_file_location(
    "rng_install_report", ROOT / "benches" / "rng_install_report.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
BENCHES = str(ROOT / "benches")
sys.path.insert(0, BENCHES)
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.remove(BENCHES)


def write_report(path: Path, core: int, full_seeded_ns: float = 5_000.0) -> None:
    """Write one complete deterministic fixture report."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for key, value in {
            "core": core,
            "governor": "performance",
            "python": "runtime-version-from-record",
            "torch": "torch-version-from-record",
            "numpy": "numpy-version-from-record",
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
            value = full_seeded_ns if metric == "full_seeded_sample" else 1_000.0
            for trial in range(10):
                writer.writerow(
                    ("data", metric, trial, 100, int(value * 100), value, trial + 1)
                )


class RngInstallReportTest(unittest.TestCase):
    """Exercise complete reports and named invalid assumptions."""

    def test_reference_upper_bound_decides_pass_and_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            performance = root / "performance.csv"
            efficiency = root / "efficiency.csv"
            write_report(performance, 19)
            write_report(efficiency, 0)
            self.assertEqual(
                MODULE.evaluate(performance, efficiency, 19, 0)["decision"], "PASS"
            )
            write_report(performance, 19, full_seeded_ns=7_000.0)
            self.assertEqual(
                MODULE.evaluate(performance, efficiency, 19, 0)["decision"], "FAIL"
            )

    def test_wrong_core_and_incomplete_metrics_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.csv"
            write_report(path, 19)
            with self.assertRaisesRegex(ValueError, "expected core"):
                MODULE.read_report(path, 18)
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text(
                "\n".join(line for line in lines if ",full_seeded_sample," not in line)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "complete metric set"):
                MODULE.read_report(path, 19)


if __name__ == "__main__":
    unittest.main()

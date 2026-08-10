"""Tests for the native RNG floor measurement verifier."""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).parents[2]


def _load_analyzer() -> ModuleType:
    path = ROOT / "benches" / "rng_floor.py"
    spec = importlib.util.spec_from_file_location("rng_floor_analyzer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analyzer = _load_analyzer()


def _write_report(
    path: Path,
    label: str,
    core: int,
    sample_ns: float = 20.0,
    governor: str = "performance",
    trials: int = 10,
) -> None:
    metadata = {
        "label": label,
        "core": str(core),
        "cpu_model": "Synthetic Spark core",
        "governor": governor,
        "max_freq_khz": "4000000" if label == "perf" else "2800000",
        "trials": str(trials),
        "iterations": "100000",
        "warmup_iterations": "10000",
        "sample_derivation_blocks": "1",
        "state_synthesis_blocks": "312",
        "feistel_rounds": "8",
    }
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for key, value in metadata.items():
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
                "freq_khz",
            )
        )
        for trial in range(trials):
            for metric in sorted(analyzer.EXPECTED_METRICS):
                value = sample_ns if metric == "sample_derivation" else 100.0
                writer.writerow(
                    (
                        "data",
                        metric,
                        trial,
                        100000,
                        int(value * 100000),
                        value,
                        trial + 1,
                        metadata["max_freq_khz"],
                    )
                )


class RngFloorHarnessTest(unittest.TestCase):
    """Exercise decisions and every named measurement assumption."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        directory = Path(self.temporary.name)
        self.performance_path = directory / "perf.csv"
        self.efficiency_path = directory / "eff.csv"
        _write_report(self.performance_path, "perf", 19)
        _write_report(self.efficiency_path, "eff", 0, sample_ns=30.0)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_valid_reports_pass_and_include_cluster_ratios(self) -> None:
        result = analyzer.evaluate_reports(
            analyzer.parse_report(self.performance_path),
            analyzer.parse_report(self.efficiency_path),
            19,
            0,
        )
        self.assertEqual(result["decision"], "PASS")
        self.assertEqual(
            result["efficiency_to_performance_ratio"]["sample_derivation"], 1.5
        )

    def test_sample_bound_failure_is_red(self) -> None:
        _write_report(self.performance_path, "perf", 19, sample_ns=26.0)
        result = analyzer.evaluate_reports(
            analyzer.parse_report(self.performance_path),
            analyzer.parse_report(self.efficiency_path),
            19,
            0,
        )
        self.assertEqual(result["decision"], "FAIL")

    def test_nonperformance_governor_is_rejected(self) -> None:
        _write_report(self.performance_path, "perf", 19, governor="powersave")
        with self.assertRaisesRegex(ValueError, "governor"):
            analyzer.validate_report(
                analyzer.parse_report(self.performance_path), "perf", 19
            )

    def test_wrong_core_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "wrong cluster or core"):
            analyzer.validate_report(
                analyzer.parse_report(self.performance_path), "perf", 18
            )

    def test_missing_trials_are_rejected(self) -> None:
        _write_report(self.performance_path, "perf", 19, trials=9)
        with self.assertRaisesRegex(ValueError, "repetition count"):
            analyzer.validate_report(
                analyzer.parse_report(self.performance_path), "perf", 19
            )

    def test_frequency_drop_is_rejected(self) -> None:
        report = analyzer.parse_report(self.performance_path)
        changed = analyzer.Report(
            report.metadata,
            report.measurements,
            [1, *report.frequencies[1:]],
            report.checksums,
        )
        with self.assertRaisesRegex(ValueError, "pinned window"):
            analyzer.validate_report(changed, "perf", 19)

    def test_contract_metadata_assumptions_are_rejected(self) -> None:
        report = analyzer.parse_report(self.performance_path)
        cases = (
            ("sample_derivation_blocks", "2", "one Philox block"),
            ("state_synthesis_blocks", "311", "312 Philox blocks"),
            ("feistel_rounds", "7", "eight Feistel rounds"),
            ("iterations", "99999", "repetition count"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                metadata = dict(report.metadata)
                metadata[field] = value
                changed = analyzer.Report(
                    metadata,
                    report.measurements,
                    report.frequencies,
                    report.checksums,
                )
                with self.assertRaisesRegex(ValueError, message):
                    analyzer.validate_report(changed, "perf", 19)

    def test_incomplete_metric_set_is_rejected(self) -> None:
        report = analyzer.parse_report(self.performance_path)
        measurements = dict(report.measurements)
        measurements.pop("native_draw")
        changed = analyzer.Report(
            report.metadata, measurements, report.frequencies, report.checksums
        )
        with self.assertRaisesRegex(ValueError, "every required operation"):
            analyzer.validate_report(changed, "perf", 19)

    def test_zero_checksums_are_rejected(self) -> None:
        report = analyzer.parse_report(self.performance_path)
        changed = analyzer.Report(
            report.metadata,
            report.measurements,
            report.frequencies,
            [0] * len(report.checksums),
        )
        with self.assertRaisesRegex(ValueError, "observable work"):
            analyzer.validate_report(changed, "perf", 19)

    def test_planted_metric_mutation_changes_the_decision(self) -> None:
        previous = os.environ.get("HYPERLOADER_RNG_FLOOR_MUTATION")
        os.environ["HYPERLOADER_RNG_FLOOR_MUTATION"] = "inflate-sample"
        try:
            result = analyzer.evaluate_reports(
                analyzer.parse_report(self.performance_path),
                analyzer.parse_report(self.efficiency_path),
                19,
                0,
            )
        finally:
            if previous is None:
                os.environ.pop("HYPERLOADER_RNG_FLOOR_MUTATION", None)
            else:
                os.environ["HYPERLOADER_RNG_FLOOR_MUTATION"] = previous
        self.assertEqual(result["decision"], "FAIL")


if __name__ == "__main__":
    unittest.main()

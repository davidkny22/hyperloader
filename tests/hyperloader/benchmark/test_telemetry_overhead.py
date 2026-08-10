"""Tests for the telemetry overhead measurement verifier."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).parents[3]


def _load_report_module() -> ModuleType:
    path = ROOT / "benches" / "telemetry_overhead_report.py"
    spec = importlib.util.spec_from_file_location("telemetry_overhead_report_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


report_module = _load_report_module()


def _report(*, telemetry_penalty: float = 0.0, cpu_ns_per_batch: float = 100.0) -> dict[str, object]:
    batches = 4096
    baseline_wall = 100_000_000
    baseline_cpu = 90_000_000
    telemetry_pairs = []
    noise_pairs = []
    for index in range(20):
        enabled_wall = round(baseline_wall * (1.0 + telemetry_penalty))
        enabled_cpu = round(baseline_cpu + cpu_ns_per_batch * batches)
        disabled = {
            "checksum": 123,
            "cpu_ns": baseline_cpu,
            "wall_ns": baseline_wall,
        }
        enabled = {
            "checksum": 123,
            "cpu_ns": enabled_cpu,
            "wall_ns": enabled_wall,
        }
        left, right = (enabled, disabled) if index % 2 == 0 else (disabled, enabled)
        telemetry_pairs.append(
            {
                "left_checksum": left["checksum"],
                "left_cpu_ns": left["cpu_ns"],
                "left_wall_ns": left["wall_ns"],
                "order": "enabled-first" if index % 2 == 0 else "disabled-first",
                "right_checksum": right["checksum"],
                "right_cpu_ns": right["cpu_ns"],
                "right_wall_ns": right["wall_ns"],
            }
        )
        noise_delta = 50_000 if index % 2 == 0 else -50_000
        noise_pairs.append(
            {
                "left_checksum": 123,
                "left_cpu_ns": baseline_cpu,
                "left_wall_ns": baseline_wall + noise_delta,
                "order": "null",
                "right_checksum": 123,
                "right_cpu_ns": baseline_cpu,
                "right_wall_ns": baseline_wall,
            }
        )
    return {
        "metadata": {
            "batch_size": 64,
            "cpu_batches_per_half": batches,
            "extension_path": "/installed/hyperloader/_hyperloader.so",
            "pair_count": 20,
            "pace_ns": 800_000,
            "platform": "test-platform",
            "process_clock_resolution_ns": 100,
            "public_path_verified": True,
            "python": "3.12.test",
            "target_sample_rate": 80_000,
            "telemetry_summary_verified": True,
            "torch": "2.test",
            "wall_batches_per_half": 128,
        },
        "cpu_pairs": [dict(pair) for pair in telemetry_pairs],
        "noise_pairs": noise_pairs,
        "wall_pairs": telemetry_pairs,
    }


class TelemetryOverheadReportTest(unittest.TestCase):
    """Exercise both decision bounds and every named report assumption."""

    def test_below_noise_and_cpu_budget_pass(self) -> None:
        result = report_module.evaluate_report(_report())
        self.assertEqual(result["decision"], "PASS")
        self.assertTrue(result["wall"]["below_noise"])
        self.assertTrue(result["cpu"]["within_budget"])

    def test_detectable_wall_penalty_fails(self) -> None:
        result = report_module.evaluate_report(_report(telemetry_penalty=0.01))
        self.assertEqual(result["decision"], "FAIL")
        self.assertFalse(result["wall"]["below_noise"])

    def test_absolute_cpu_budget_failure_is_red(self) -> None:
        result = report_module.evaluate_report(_report(cpu_ns_per_batch=1_000.0))
        self.assertEqual(result["decision"], "FAIL")
        self.assertFalse(result["cpu"]["within_budget"])

    def test_named_measurement_assumptions_are_rejected(self) -> None:
        cases = (
            ("pair_count", 19, "pair count"),
            ("batch_size", 32, "batch size"),
            ("target_sample_rate", 79_999, "sample rate"),
            ("pace_ns", 799_999, "pacing"),
            ("public_path_verified", False, "installed artifact"),
            ("telemetry_summary_verified", False, "epoch summary"),
            ("process_clock_resolution_ns", 10_000, "clock resolution"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                report = _report()
                report["metadata"][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    report_module.validate_report(report)

    def test_nonidentical_delivery_is_rejected(self) -> None:
        report = _report()
        report["cpu_pairs"][0]["right_checksum"] = 456
        with self.assertRaisesRegex(ValueError, "identical values"):
            report_module.validate_report(report)

    def test_nonalternating_order_is_rejected(self) -> None:
        report = _report()
        report["cpu_pairs"][1]["order"] = "enabled-first"
        with self.assertRaisesRegex(ValueError, "alternate"):
            report_module.validate_report(report)

    def test_short_half_is_rejected(self) -> None:
        report = _report()
        report["noise_pairs"][0]["left_wall_ns"] = 1
        with self.assertRaisesRegex(ValueError, "too short"):
            report_module.validate_report(report)

    def test_planted_cost_mutation_changes_the_decision(self) -> None:
        previous = os.environ.get("HYPERLOADER_TELEMETRY_MUTATION")
        os.environ["HYPERLOADER_TELEMETRY_MUTATION"] = "inflate-cost"
        try:
            result = report_module.evaluate_report(_report())
        finally:
            if previous is None:
                os.environ.pop("HYPERLOADER_TELEMETRY_MUTATION", None)
            else:
                os.environ["HYPERLOADER_TELEMETRY_MUTATION"] = previous
        self.assertEqual(result["decision"], "FAIL")


if __name__ == "__main__":
    unittest.main()

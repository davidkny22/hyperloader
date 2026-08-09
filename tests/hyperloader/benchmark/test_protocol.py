"""Paired benchmark matrix, validation, metadata, and decision tests."""

from __future__ import annotations

import unittest
from dataclasses import replace

from benches.benchmark_protocol import (
    WORKLOAD_MATRIX,
    CommonConfig,
    EnvironmentMetadata,
    PairedObservation,
    ProtocolError,
    SystemRun,
    TuningBudget,
    capture_environment,
    evaluate,
    validate_observations,
)


def environment() -> EnvironmentMetadata:
    """Return pinned synthetic metadata for protocol checks."""
    return EnvironmentMetadata(
        captured_at="2026-08-09T00:00:00+00:00",
        machine="spark",
        operating_system="Linux",
        kernel="6.11",
        architecture="aarch64",
        python="3.12.3",
        commit="0123456",
        cpu_governor="performance",
        gpu_clock="locked-2500MHz",
        cache_regime="warm",
        benchmark_mode=True,
        concurrent_load=False,
    )


def observations(count: int, penalties: list[float] | None = None) -> list[PairedObservation]:
    """Build valid alternating paired observations."""
    metadata = environment()
    config = CommonConfig(
        workload="fixed-text",
        gpu_regime="compute",
        batch_size=64,
        workers=4,
        prefetch_depth=128,
        delivery="host",
        batch_shape="int64[64,512]",
        cache_regime="warm",
    )
    tuning = TuningBudget(8, 600.0, ("workers", "prefetch_depth"))
    values = penalties or [0.5] * count
    records = []
    for ordinal in range(count):
        reference = SystemRun(
            "counterfactual", 1000.0, 45.0, True, config, tuning, metadata
        )
        loader = SystemRun(
            "loader",
            1000.0 * (1.0 - values[ordinal] / 100.0),
            45.0,
            True,
            config,
            tuning,
            metadata,
        )
        first, second = (reference, loader) if ordinal % 2 == 0 else (loader, reference)
        records.append(PairedObservation(ordinal, first, second, True))
    return records


class BenchmarkProtocolTest(unittest.TestCase):
    """Exercise every fixed protocol invariant and terminal branch."""

    def test_matrix_contains_the_six_fixed_cells(self) -> None:
        self.assertEqual(
            [workload.name for workload in WORKLOAD_MATRIX],
            [
                "images-light",
                "images-heavy",
                "fixed-text",
                "varlen-text",
                "arrow-tabular",
                "numpy-array",
            ],
        )
        self.assertEqual(sum(cell.transport_bound for cell in WORKLOAD_MATRIX), 3)

    def test_environment_capture_records_host_and_controls(self) -> None:
        captured = capture_environment(
            commit="abcdef0",
            cpu_governor="performance",
            gpu_clock="locked",
            cache_regime="warm",
            benchmark_mode=True,
            concurrent_load=False,
        )
        self.assertTrue(captured.machine)
        self.assertTrue(captured.operating_system)
        self.assertTrue(captured.architecture)
        self.assertEqual(captured.commit, "abcdef0")

    def test_minimum_replication_and_upper_bound_decide(self) -> None:
        self.assertEqual(
            evaluate(observations(9), threshold_percent=1.0).status, "collect"
        )
        passed = evaluate(observations(10), threshold_percent=1.0)
        self.assertEqual(passed.status, "pass")
        self.assertAlmostEqual(passed.mean_penalty_percent, 0.5)
        self.assertAlmostEqual(passed.half_width_percent, 0.0)
        failed = evaluate(observations(10, [1.2] * 10), threshold_percent=1.0)
        self.assertEqual(failed.status, "fail")

    def test_precision_rule_collects_until_cap(self) -> None:
        noisy = [0.0, 1.8] * 5
        self.assertEqual(
            evaluate(observations(10, noisy), threshold_percent=2.0).status,
            "collect",
        )
        capped = evaluate(observations(40, [0.0, 1.8] * 20), threshold_percent=2.0)
        self.assertIn(capped.status, {"pass", "fail"})
        self.assertGreater(capped.half_width_percent, 0.15)

    def test_invalid_order_and_equal_tuning_are_rejected(self) -> None:
        records = observations(10)
        records[1] = replace(
            records[1], first=records[1].second, second=records[1].first
        )
        with self.assertRaisesRegex(ProtocolError, "alternate"):
            validate_observations(records)

        records = observations(10)
        changed = replace(records[0].second.tuning, trials=9)
        records[0] = replace(
            records[0], second=replace(records[0].second, tuning=changed)
        )
        with self.assertRaisesRegex(ProtocolError, "equal counted tuning"):
            validate_observations(records)

        records = observations(10)
        changed_config = replace(records[1].first.config, batch_size=32)
        records[1] = replace(
            records[1],
            first=replace(records[1].first, config=changed_config),
            second=replace(records[1].second, config=changed_config),
        )
        with self.assertRaisesRegex(ProtocolError, "mix common configurations"):
            validate_observations(records)

    def test_unpinned_or_interrupted_records_are_rejected(self) -> None:
        records = observations(10)
        unpinned = replace(records[0].first.environment, gpu_clock="")
        records[0] = replace(
            records[0],
            first=replace(records[0].first, environment=unpinned),
            second=replace(records[0].second, environment=unpinned),
        )
        with self.assertRaisesRegex(ProtocolError, "clock and governor"):
            validate_observations(records)

        records = observations(10)
        records[0] = replace(records[0], uninterrupted=False)
        with self.assertRaisesRegex(ProtocolError, "uninterrupted"):
            validate_observations(records)

    def test_fixed_configuration_accepts_only_exact_zero_tuning(self) -> None:
        records = observations(10)
        fixed = TuningBudget(0, 0.0, ())
        records = [
            replace(
                record,
                first=replace(record.first, tuning=fixed),
                second=replace(record.second, tuning=fixed),
            )
            for record in records
        ]
        validate_observations(records)

        partial = replace(fixed, trials=1)
        records[0] = replace(
            records[0],
            first=replace(records[0].first, tuning=partial),
            second=replace(records[0].second, tuning=partial),
        )
        with self.assertRaisesRegex(ProtocolError, "exact zero or fully positive"):
            validate_observations(records)


if __name__ == "__main__":
    unittest.main()

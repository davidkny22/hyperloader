"""Independent full-campaign overhead report checks."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from benches.benchmark_protocol import (
    CommonConfig,
    EnvironmentMetadata,
    PairedObservation,
    SystemRun,
    TuningBudget,
)

BENCHES = Path(__file__).parents[3] / "benches"
sys.path.insert(0, str(BENCHES))
import overhead_report  # noqa: E402


class OverheadReportTest(TestCase):
    """Require terminal reproduction, protocol controls, and byte ceilings."""

    def _cell(self, ordinal: int) -> dict[str, object]:
        environment = EnvironmentMetadata(
            captured_at="captured-at-from-record",
            machine="machine-under-test",
            operating_system="operating-system-under-test",
            kernel="kernel-from-record",
            architecture="architecture-under-test",
            python="runtime-version-from-record",
            commit="source-revision-from-record",
            cpu_governor="governor-from-record",
            gpu_clock="clock-from-record",
            cache_regime="warm",
            benchmark_mode=True,
            concurrent_load=False,
        )
        first_name, second_name = (
            ("counterfactual", "loader")
            if ordinal % 2 == 0
            else ("loader", "counterfactual")
        )
        throughput = {"counterfactual": 100.0, "loader": 99.8}
        config = CommonConfig(
            workload="fixed-text",
            gpu_regime="compute",
            batch_size=64,
            workers=4,
            prefetch_depth=128,
            delivery="host-sync-h2d",
            batch_shape="int64[64,512]",
            cache_regime="warm",
        )
        tuning = TuningBudget(trials=0, wall_seconds=0.0, knobs=())
        run = lambda name: SystemRun(  # noqa: E731
            system=name,
            throughput=throughput[name],
            duration_seconds=45.0,
            warmed=True,
            config=config,
            tuning=tuning,
            environment=environment,
        )
        cell = asdict(
            PairedObservation(
                ordinal=ordinal,
                first=run(first_name),
                second=run(second_name),
                uninterrupted=True,
            )
        )
        cell["raw"] = {
            "clock_samples": [{"clock_mhz": 997, "utilization_percent": 99}],
            "llc_bytes": 24,
            "resident_bytes": 192,
            "byte_split": {
                "model_input_gbps": 0.1,
                "irreducible_host_gbps": 0.0,
                "explicit_overhead_gbps": 0.0,
                "explicit_total_host_gbps": 0.0,
            },
        }
        return cell

    def test_regime_reproduces_terminal_decision(self) -> None:
        cells = [self._cell(ordinal) for ordinal in range(10)]
        decision = asdict(
            overhead_report.evaluate(
                [overhead_report.decode_observation(cell) for cell in cells],
                threshold_percent=2.0,
            )
        )
        with TemporaryDirectory() as directory:
            campaign = Path(directory)
            (campaign / "compute-cells.jsonl").write_text(
                "".join(json.dumps(cell) + "\n" for cell in cells), encoding="utf-8"
            )
            (campaign / "compute-decision.json").write_text(
                json.dumps(decision), encoding="utf-8"
            )
            report = overhead_report.verify_regime(campaign, "compute")
        self.assertEqual(report["cells"], 10)
        self.assertLess(report["decision"]["upper_percent"], 1.0)

    def test_regime_rejects_broken_alternation(self) -> None:
        cells = [self._cell(ordinal) for ordinal in range(10)]
        cells[1]["first"], cells[1]["second"] = cells[1]["second"], cells[1]["first"]
        with (
            TemporaryDirectory() as directory,
            patch.object(overhead_report, "_read_cells", return_value=cells),
        ):
            with self.assertRaisesRegex(AssertionError, "did not alternate"):
                overhead_report.verify_regime(Path(directory), "compute")

    def test_regime_rejects_copied_byte_overrun(self) -> None:
        cells = [self._cell(ordinal) for ordinal in range(10)]
        for cell in cells:
            cell["raw"]["byte_split"]["explicit_overhead_gbps"] = 0.6
        decision = asdict(
            overhead_report.evaluate(
                [overhead_report.decode_observation(cell) for cell in cells],
                threshold_percent=2.0,
            )
        )
        with TemporaryDirectory() as directory:
            campaign = Path(directory)
            (campaign / "compute-cells.jsonl").write_text(
                "".join(json.dumps(cell) + "\n" for cell in cells), encoding="utf-8"
            )
            (campaign / "compute-decision.json").write_text(
                json.dumps(decision), encoding="utf-8"
            )
            with self.assertRaisesRegex(AssertionError, "copied-byte ceiling"):
                overhead_report.verify_regime(campaign, "compute")

    def test_control_validation_uses_the_run_record(self) -> None:
        clock_request_mhz = 997
        validated_mhz = overhead_report._validate_controls(
            {
                "gpu_clock": f"locked-{clock_request_mhz}MHz",
                "cpu_governor": "governor-from-run",
            },
            {
                "requested_mhz": clock_request_mhz,
                "command_returncode": 0,
                "reset_stdout": "reset-complete",
            },
            expected_cpu_governor="governor-from-run",
        )
        self.assertEqual(validated_mhz, clock_request_mhz)

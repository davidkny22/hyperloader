"""Installed public gate for causal controller ceiling reports."""

from __future__ import annotations

import os
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from hyperloader import DataLoader, HyperConfig, _hyperloader
from hyperloader.config import CeilingConfig, ControlConfig, FactorConfig
from hyperloader.control.runtime import decision_report as _decision_report
from hyperloader.process.frontier import FrontierRuntime


def _without_binding(decision: object) -> dict[str, object]:
    report = _decision_report(decision)
    report["binding"] = None
    return report


def _run_report(*, ceilings: CeilingConfig, stalled: bool) -> dict[str, object]:
    config = HyperConfig(
        control=ControlConfig(cadence=1e-9, ceilings=ceilings),
        factors=FactorConfig(hysteresis=1),
    )
    loader = DataLoader(
        range(24),
        batch_size=1,
        num_workers=4,
        seed=419,
        config=config,
    )
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                FrontierRuntime, "consume_stall_flag", return_value=stalled
            )
        )
        stack.enter_context(
            mock.patch.object(
                FrontierRuntime,
                "occupied",
                new_callable=mock.PropertyMock,
                return_value=8,
            )
        )
        if os.environ.get("HYPERLOADER_CEILING_MUTATION") == "drop-binding":
            stack.enter_context(
                mock.patch("hyperloader.control.decision_report", _without_binding)
            )
        try:
            stream = [int(batch.item()) for batch in loader]
            report = loader.stats()["controller"]
        finally:
            loader.close()
    if stream != list(range(24)):
        raise AssertionError("controller ceilings changed the sampler stream")
    if not isinstance(report, dict):
        raise AssertionError("controller did not publish a decision report")
    return report


class CeilingBindsReportGate(unittest.TestCase):
    """Name the user ceiling that prevents a starvation-clearing decision."""

    def test_starvation_at_cpu_ceiling_names_cpu_cores(self) -> None:
        expected_root = os.environ.get("HYPERLOADER_EXPECTED_INSTALL_ROOT")
        if expected_root is not None:
            self.assertTrue(
                Path(_hyperloader.__file__)
                .resolve()
                .is_relative_to(Path(expected_root).resolve())
            )

        report = _run_report(ceilings=CeilingConfig(cpu_cores=1), stalled=True)

        self.assertTrue(report["starvation"])
        self.assertEqual(report["binding"], "cpu_cores")

    def test_measured_bandwidth_violation_names_bandwidth(self) -> None:
        report = _run_report(ceilings=CeilingConfig(bandwidth=0.0), stalled=True)

        self.assertTrue(report["starvation"])
        self.assertEqual(report["binding"], "bandwidth")

    def test_cpu_ceiling_without_starvation_is_not_named_as_causal(self) -> None:
        report = _run_report(ceilings=CeilingConfig(cpu_cores=1), stalled=False)

        self.assertFalse(report["starvation"])
        self.assertIsNone(report["binding"])


if __name__ == "__main__":
    unittest.main()

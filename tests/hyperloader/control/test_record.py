"""Calibration curve validation and persistence checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hyperloader.control import (
    BandwidthPoint,
    CalibrationRecord,
    CpuCluster,
    IdleStateTax,
    MachineIdentity,
    PinCost,
    StagedCopyTax,
    StealCurve,
    StealPoint,
    calibration_cache_path,
    load_calibration,
    save_calibration,
)


def _record(machine: MachineIdentity) -> CalibrationRecord:
    compute = tuple(
        StealPoint(cores, loss)
        for cores, loss in zip(
            (1, 2, 4, 8, 16), (0.01, 0.015, 0.02, 0.03, 0.04), strict=True
        )
    )
    stream = tuple(
        StealPoint(cores, loss)
        for cores, loss in zip(
            (1, 2, 4, 8, 16), (0.02, 0.03, 0.04, 0.05, 0.06), strict=True
        )
    )
    return CalibrationRecord(
        machine=machine,
        source="measured fixture",
        measured_at="2026-08-10",
        steal_curves=(
            StealCurve("all", "compute", compute),
            StealCurve("all", "stream", stream),
        ),
        bandwidth_curve=(
            BandwidthPoint(0.0, 0.0, 0.0),
            BandwidthPoint(1_000_000.0, 0.001, 0.002),
        ),
        bandwidth_provenance="measured",
        spawn_nanoseconds=10,
        pin_cost=PinCost(4096, 20),
        idle_state_tax=IdleStateTax(0.1, 0.5, 0.05, 2_000_000),
        staged_copy_tax=StagedCopyTax(4096, 0.12),
    )


class CalibrationRecordTest(unittest.TestCase):
    """Exercise cache keys, round trips, and negative record assumptions."""

    def setUp(self) -> None:
        self.machine = MachineIdentity("cpu", (CpuCluster("all", (0, 1)),), 1024)

    def test_cache_round_trip_preserves_curves_and_opaque_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = calibration_cache_path(Path(directory), self.machine)
            save_calibration(_record(self.machine), path)
            loaded = load_calibration(path, self.machine)

        self.assertEqual(loaded, _record(self.machine))
        self.assertEqual(path.stem, self.machine.cache_key)

    def test_changed_machine_invalidates_cached_record(self) -> None:
        changed = MachineIdentity("cpu", self.machine.clusters, 2048)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            save_calibration(_record(self.machine), path)

            self.assertIsNone(load_calibration(path, changed))

    def test_scalar_and_unsorted_curves_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two points"):
            StealCurve("all", "compute", (), "derived-prior")
        with self.assertRaisesRegex(ValueError, "unique and increasing"):
            StealCurve(
                "all",
                "compute",
                (StealPoint(2, 0.02), StealPoint(1, 0.01)),
                "derived-prior",
            )

    def test_measured_curve_requires_the_complete_width_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "1/2/4/8/16"):
            StealCurve("all", "compute", (StealPoint(1, 0.01), StealPoint(2, 0.02)))

    def test_tax_measurements_reject_invalid_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "warm duty"):
            IdleStateTax(0.1, 0.5, 0.0, 2_000_000)
        with self.assertRaisesRegex(ValueError, "positive batch size"):
            StagedCopyTax(0, 0.1)

    def test_persisted_records_require_explicit_tax_measurements(self) -> None:
        payload = _record(self.machine).to_dict()
        del payload["idle_state_tax"]

        with self.assertRaisesRegex(ValueError, "tax measurements"):
            CalibrationRecord.from_dict(payload)


if __name__ == "__main__":
    unittest.main()

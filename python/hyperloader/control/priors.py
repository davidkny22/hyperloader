"""Measured machine-class priors used before local calibration exists."""

from __future__ import annotations

from .machine import MachineIdentity
from .record import (
    BandwidthPoint,
    CalibrationRecord,
    PinCost,
    StealCurve,
    StealPoint,
)


def spark_prior(machine: MachineIdentity) -> CalibrationRecord | None:
    """Return the Spark campaign prior only for the measured hardware class."""
    model = machine.cpu_model.casefold()
    if not (
        "nvidia grace" in model
        or ("cortex-x925" in model and "cortex-a725" in model)
    ):
        return None
    return CalibrationRecord(
        machine=machine,
        source="DGX Spark anchor campaign 2026-08-08",
        measured_at="2026-08-08",
        steal_curves=(
            StealCurve(
                "performance-first",
                "compute",
                (
                    StealPoint(1, 0.0204),
                    StealPoint(4, 0.061),
                    StealPoint(8, 0.140),
                    StealPoint(16, 0.216),
                ),
                "derived-prior",
            ),
            StealCurve(
                "efficiency",
                "compute",
                (StealPoint(1, 0.0075), StealPoint(4, 0.013)),
                "derived-prior",
            ),
            StealCurve(
                "performance-first",
                "stream",
                (
                    StealPoint(4, 0.204),
                    StealPoint(8, 0.320),
                    StealPoint(16, 0.404),
                ),
                "derived-prior",
            ),
        ),
        bandwidth_curve=(
            BandwidthPoint(0.0, 0.0, 0.0),
            BandwidthPoint(1_000_000_000.0, 0.0023, 0.0011),
        ),
        bandwidth_provenance="derived-prior",
        spawn_nanoseconds=13_800_000,
        pin_cost=PinCost(268_435_456, 9_000_000),
    )

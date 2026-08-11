"""Measured machine-class priors used before local calibration exists."""

from __future__ import annotations

from .machine import MachineIdentity
from .record import (
    BandwidthPoint,
    CalibrationRecord,
    IdleStateTax,
    PinCost,
    StagedCopyTax,
    StealCurve,
    StealPoint,
)


def spark_prior(machine: MachineIdentity) -> CalibrationRecord | None:
    """Return the Spark campaign prior only for the measured hardware class."""
    model = machine.cpu_model.casefold()
    logical_cpus = {cpu for cluster in machine.clusters for cpu in cluster.logical_cpus}
    frequencies = {
        cluster.max_frequency_hz
        for cluster in machine.clusters
        if cluster.max_frequency_hz is not None
    }
    detected_spark_topology = (
        model == "aarch64"
        and len(logical_cpus) == 20
        and 2_808_000_000 in frequencies
        and 4_004_000_000 in frequencies
        and 115 * 1024**3 <= machine.memory_bytes <= 130 * 1024**3
    )
    if not (
        "nvidia grace" in model
        or ("cortex-x925" in model and "cortex-a725" in model)
        or detected_spark_topology
    ):
        return None
    return CalibrationRecord(
        machine=machine,
        source="DGX Spark anchor and fixed-text campaigns",
        measured_at="2026-08-11",
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
        idle_state_tax=IdleStateTax(
            loss_fraction=0.23617,
            powered_down_residency_fraction=0.68708,
            warm_duty_fraction=0.05,
            minimum_gap_nanoseconds=1_930_000,
        ),
        staged_copy_tax=StagedCopyTax(
            batch_bytes=262_144,
            loss_fraction=0.1329,
        ),
    )

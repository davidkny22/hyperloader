"""Validated calibration curves and primitive cost measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .machine import MachineIdentity

CALIBRATION_SCHEMA = 2
CALIBRATION_CORE_COUNTS = (1, 2, 4, 8, 16)


@dataclass(frozen=True, slots=True)
class StealPoint:
    """Accelerator throughput loss observed at one CPU width."""

    cores: int
    loss_fraction: float

    def __post_init__(self) -> None:
        if self.cores <= 0:
            raise ValueError("steal-curve core count must be positive")
        _validate_loss(self.loss_fraction)


@dataclass(frozen=True, slots=True)
class StealCurve:
    """Width-to-loss curve for one cluster and loader work shape."""

    cluster: str
    work_shape: str
    points: tuple[StealPoint, ...]
    provenance: str = "measured"

    def __post_init__(self) -> None:
        if not self.cluster or self.work_shape not in {"compute", "stream"}:
            raise ValueError("steal curves require a cluster and known work shape")
        _validate_points(self.points, lambda point: point.cores)
        if self.provenance not in {"measured", "derived-prior"}:
            raise ValueError("calibration provenance must be measured or derived-prior")
        if (
            self.provenance == "measured"
            and tuple(point.cores for point in self.points) != CALIBRATION_CORE_COUNTS
        ):
            raise ValueError("measured steal curves require the 1/2/4/8/16 core grid")


@dataclass(frozen=True, slots=True)
class BandwidthPoint:
    """Accelerator loss at one loader-attributable copied-byte rate."""

    bytes_per_second: float
    compute_loss_fraction: float
    bandwidth_loss_fraction: float

    def __post_init__(self) -> None:
        if self.bytes_per_second < 0:
            raise ValueError("bandwidth rate must be nonnegative")
        _validate_loss(self.compute_loss_fraction)
        _validate_loss(self.bandwidth_loss_fraction)


@dataclass(frozen=True, slots=True)
class PinCost:
    """One measured host-pinning registration cost."""

    bytes: int
    nanoseconds: int

    def __post_init__(self) -> None:
        if self.bytes <= 0 or self.nanoseconds <= 0:
            raise ValueError("pin cost requires positive bytes and nanoseconds")


@dataclass(frozen=True, slots=True)
class IdleStateTax:
    """Measured quiet-consumer loss and the duty that removed idle entries."""

    loss_fraction: float
    powered_down_residency_fraction: float
    warm_duty_fraction: float
    minimum_gap_nanoseconds: int

    def __post_init__(self) -> None:
        _validate_loss(self.loss_fraction)
        _validate_fraction(
            "powered-down residency", self.powered_down_residency_fraction
        )
        _validate_positive_fraction("warm duty", self.warm_duty_fraction)
        if self.minimum_gap_nanoseconds <= 0:
            raise ValueError("idle-state tax requires a positive minimum gap")


@dataclass(frozen=True, slots=True)
class StagedCopyTax:
    """Measured pageable-to-pinned delivery loss for one batch shape."""

    batch_bytes: int
    loss_fraction: float

    def __post_init__(self) -> None:
        if self.batch_bytes <= 0:
            raise ValueError("staged-copy tax requires a positive batch size")
        _validate_positive_fraction("staged-copy loss", self.loss_fraction)


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    """Machine-keyed curves used by the resource controller."""

    machine: MachineIdentity
    source: str
    measured_at: str
    steal_curves: tuple[StealCurve, ...]
    bandwidth_curve: tuple[BandwidthPoint, ...]
    bandwidth_provenance: str
    spawn_nanoseconds: int
    pin_cost: PinCost
    idle_state_tax: IdleStateTax | None
    staged_copy_tax: StagedCopyTax | None
    schema: int = CALIBRATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CALIBRATION_SCHEMA:
            raise ValueError("calibration schema is unsupported")
        if not self.source or not self.measured_at:
            raise ValueError("calibration source and measurement time are required")
        if self.spawn_nanoseconds <= 0:
            raise ValueError("spawn cost must be positive")
        if not self.steal_curves or not self.bandwidth_curve:
            raise ValueError("calibration records require compute and bandwidth curves")
        keys = [(curve.cluster, curve.work_shape) for curve in self.steal_curves]
        if len(keys) != len(set(keys)):
            raise ValueError("calibration steal curves must have unique keys")
        if {curve.work_shape for curve in self.steal_curves} != {"compute", "stream"}:
            raise ValueError(
                "calibration records require compute and stream steal curves"
            )
        _validate_points(self.bandwidth_curve, lambda point: point.bytes_per_second)
        if self.bandwidth_provenance not in {"measured", "derived-prior"}:
            raise ValueError("bandwidth provenance must be measured or derived-prior")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible payload."""
        return {
            "bandwidth_curve": [
                {
                    "bandwidth_loss_fraction": point.bandwidth_loss_fraction,
                    "bytes_per_second": point.bytes_per_second,
                    "compute_loss_fraction": point.compute_loss_fraction,
                }
                for point in self.bandwidth_curve
            ],
            "bandwidth_provenance": self.bandwidth_provenance,
            "machine": self.machine.to_dict(),
            "machine_key": self.machine.cache_key,
            "measured_at": self.measured_at,
            "pin_cost": {
                "bytes": self.pin_cost.bytes,
                "nanoseconds": self.pin_cost.nanoseconds,
            },
            "idle_state_tax": (
                None
                if self.idle_state_tax is None
                else {
                    "loss_fraction": self.idle_state_tax.loss_fraction,
                    "minimum_gap_nanoseconds": self.idle_state_tax.minimum_gap_nanoseconds,
                    "powered_down_residency_fraction": (
                        self.idle_state_tax.powered_down_residency_fraction
                    ),
                    "warm_duty_fraction": self.idle_state_tax.warm_duty_fraction,
                }
            ),
            "schema": self.schema,
            "source": self.source,
            "spawn_nanoseconds": self.spawn_nanoseconds,
            "staged_copy_tax": (
                None
                if self.staged_copy_tax is None
                else {
                    "batch_bytes": self.staged_copy_tax.batch_bytes,
                    "loss_fraction": self.staged_copy_tax.loss_fraction,
                }
            ),
            "steal_curves": [
                {
                    "cluster": curve.cluster,
                    "points": [
                        {"cores": point.cores, "loss_fraction": point.loss_fraction}
                        for point in curve.points
                    ],
                    "provenance": curve.provenance,
                    "work_shape": curve.work_shape,
                }
                for curve in self.steal_curves
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationRecord:
        """Validate and construct a persisted calibration payload."""
        machine_payload = payload.get("machine")
        if not isinstance(machine_payload, dict):
            raise ValueError("calibration machine must be an object")
        machine = MachineIdentity.from_dict(machine_payload)
        if payload.get("machine_key") != machine.cache_key:
            raise ValueError("calibration machine key does not match its payload")
        raw_curves = payload.get("steal_curves")
        raw_bandwidth = payload.get("bandwidth_curve")
        if not isinstance(raw_curves, list) or not isinstance(raw_bandwidth, list):
            raise ValueError("calibration curves must be lists")
        curves = tuple(
            StealCurve(
                cluster=str(raw["cluster"]),
                work_shape=str(raw["work_shape"]),
                points=tuple(
                    StealPoint(int(point["cores"]), float(point["loss_fraction"]))
                    for point in raw["points"]
                ),
                provenance=str(raw["provenance"]),
            )
            for raw in raw_curves
        )
        bandwidth = tuple(
            BandwidthPoint(
                float(point["bytes_per_second"]),
                float(point["compute_loss_fraction"]),
                float(point["bandwidth_loss_fraction"]),
            )
            for point in raw_bandwidth
        )
        raw_pin = payload.get("pin_cost")
        if not isinstance(raw_pin, dict):
            raise ValueError("calibration pin cost must be an object")
        if "idle_state_tax" not in payload or "staged_copy_tax" not in payload:
            raise ValueError("calibration tax measurements are required")
        raw_idle = payload["idle_state_tax"]
        if raw_idle is not None and not isinstance(raw_idle, dict):
            raise ValueError("calibration idle-state tax must be an object or null")
        raw_staged = payload["staged_copy_tax"]
        if raw_staged is not None and not isinstance(raw_staged, dict):
            raise ValueError("calibration staged-copy tax must be an object or null")
        return cls(
            machine=machine,
            source=str(payload.get("source", "")),
            measured_at=str(payload.get("measured_at", "")),
            steal_curves=curves,
            bandwidth_curve=bandwidth,
            bandwidth_provenance=str(payload.get("bandwidth_provenance", "")),
            spawn_nanoseconds=int(payload.get("spawn_nanoseconds", 0)),
            pin_cost=PinCost(
                int(raw_pin.get("bytes", 0)), int(raw_pin.get("nanoseconds", 0))
            ),
            idle_state_tax=(
                None
                if raw_idle is None
                else IdleStateTax(
                    float(raw_idle.get("loss_fraction", -1.0)),
                    float(raw_idle.get("powered_down_residency_fraction", -1.0)),
                    float(raw_idle.get("warm_duty_fraction", -1.0)),
                    int(raw_idle.get("minimum_gap_nanoseconds", 0)),
                )
            ),
            staged_copy_tax=(
                None
                if raw_staged is None
                else StagedCopyTax(
                    int(raw_staged.get("batch_bytes", 0)),
                    float(raw_staged.get("loss_fraction", -1.0)),
                )
            ),
            schema=int(payload.get("schema", 0)),
        )


def _validate_loss(value: float) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError("throughput loss fraction must be in [0, 1)")


def _validate_fraction(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} fraction must be in [0, 1]")


def _validate_positive_fraction(name: str, value: float) -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} fraction must be in (0, 1]")


def _validate_points(points: tuple[Any, ...], coordinate: Any) -> None:
    if len(points) < 2:
        raise ValueError("calibration curves require at least two points")
    coordinates = [coordinate(point) for point in points]
    if coordinates != sorted(set(coordinates)):
        raise ValueError("calibration curve coordinates must be unique and increasing")

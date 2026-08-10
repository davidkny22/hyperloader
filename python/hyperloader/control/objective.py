"""Lexicographic starvation and resource-loss objective."""

from __future__ import annotations

from .record import CalibrationRecord


class ControllerObjective:
    """Evaluate starvation first and calibrated resource loss second."""

    def __init__(self, calibration: CalibrationRecord | None) -> None:
        self._calibration = calibration

    def score(
        self,
        *,
        starvation: bool,
        width: int,
        work_shape: str,
        cluster: str,
        bytes_per_second: float,
        memory_penalty: float = 0.0,
    ) -> tuple[int, float]:
        """Return a tuple whose ordinary ordering implements the objective."""
        if width <= 0 or bytes_per_second < 0 or memory_penalty < 0:
            raise ValueError("controller objective inputs must be nonnegative")
        compute_loss = self._steal_loss(width, work_shape, cluster)
        bandwidth_loss = self._bandwidth_loss(bytes_per_second, work_shape)
        return (int(starvation), compute_loss + bandwidth_loss + memory_penalty)

    def _steal_loss(self, width: int, work_shape: str, cluster: str) -> float:
        if self._calibration is None:
            return float(width)
        curve = next(
            (
                item
                for item in self._calibration.steal_curves
                if item.cluster == cluster and item.work_shape == work_shape
            ),
            None,
        )
        if curve is None:
            return float(width)
        return _interpolate(
            tuple((float(point.cores), point.loss_fraction) for point in curve.points),
            float(width),
        )

    def _bandwidth_loss(self, rate: float, work_shape: str) -> float:
        if self._calibration is None:
            return 0.0
        attribute = (
            "compute_loss_fraction"
            if work_shape == "compute"
            else "bandwidth_loss_fraction"
        )
        return _interpolate(
            tuple(
                (point.bytes_per_second, getattr(point, attribute))
                for point in self._calibration.bandwidth_curve
            ),
            rate,
        )


def _interpolate(points: tuple[tuple[float, float], ...], coordinate: float) -> float:
    if coordinate <= points[0][0]:
        return points[0][1]
    for left, right in zip(points, points[1:], strict=False):
        if coordinate <= right[0]:
            span = right[0] - left[0]
            weight = (coordinate - left[0]) / span
            return left[1] + weight * (right[1] - left[1])
    return points[-1][1]

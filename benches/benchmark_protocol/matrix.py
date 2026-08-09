"""The fixed workload cells used for loader comparisons."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Workload:
    """One benchmark cell with its output and dominant work regime."""

    name: str
    family: str
    gpu_regime: str
    variable_shape: bool
    transport_bound: bool


WORKLOAD_MATRIX = (
    Workload("images-light", "image", "compute", False, False),
    Workload("images-heavy", "image", "compute", False, False),
    Workload("fixed-text", "text", "compute", False, True),
    Workload("varlen-text", "text", "compute", True, False),
    Workload("arrow-tabular", "tabular", "bandwidth", False, True),
    Workload("numpy-array", "array", "bandwidth", False, True),
)


def workload_names() -> tuple[str, ...]:
    """Return matrix names in their fixed reporting order."""
    return tuple(workload.name for workload in WORKLOAD_MATRIX)

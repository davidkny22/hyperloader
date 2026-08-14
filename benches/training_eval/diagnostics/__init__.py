"""Residual-gap attribution for live training cells."""

from .cpu_activity import diff_cpu_activity, snapshot_cpu_activity
from .segments import DiagnosticStep, summarize_timings

__all__ = [
    "DiagnosticStep",
    "diff_cpu_activity",
    "snapshot_cpu_activity",
    "summarize_timings",
]

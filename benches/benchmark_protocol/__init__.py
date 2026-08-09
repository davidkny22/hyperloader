"""Reproducible paired benchmark records and decisions."""

from .accounting import ByteSplit, process_transport_split
from .decision import DecisionResult, evaluate
from .environment import capture_environment
from .matrix import WORKLOAD_MATRIX, Workload
from .models import (
    CommonConfig,
    EnvironmentMetadata,
    PairedObservation,
    SystemRun,
    TuningBudget,
)
from .validation import ProtocolError, validate_observations

__all__ = [
    "WORKLOAD_MATRIX",
    "ByteSplit",
    "CommonConfig",
    "DecisionResult",
    "EnvironmentMetadata",
    "PairedObservation",
    "ProtocolError",
    "SystemRun",
    "TuningBudget",
    "Workload",
    "capture_environment",
    "evaluate",
    "process_transport_split",
    "validate_observations",
]

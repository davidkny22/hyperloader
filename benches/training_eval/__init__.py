"""Machine-readable live-training evaluation protocol."""

from .ambient import AmbientDecision, AmbientProbe, compare_ambient
from .decision import TrainingDecision, decide
from .lease import FileLease, LeaseRecord, LeaseUnavailable
from .models import (
    DecisionRule,
    TrainingCellConfig,
    TrainingEnvironment,
    TrainingHalf,
    TrainingObservation,
)
from .output import write_result
from .validation import TrainingProtocolError, validate_observations

__all__ = [
    "AmbientDecision",
    "AmbientProbe",
    "DecisionRule",
    "FileLease",
    "LeaseRecord",
    "LeaseUnavailable",
    "TrainingCellConfig",
    "TrainingDecision",
    "TrainingEnvironment",
    "TrainingHalf",
    "TrainingObservation",
    "TrainingProtocolError",
    "compare_ambient",
    "decide",
    "validate_observations",
    "write_result",
]

"""Controlled-variable evidence for paired training cells."""

from .capture import ControlledVariableRecorder
from .registry import CONTROLLED_VARIABLE_REGISTRY, ControlledVariableDefinition
from .worker_probe import WorkerEnvironmentProbe

__all__ = [
    "CONTROLLED_VARIABLE_REGISTRY",
    "ControlledVariableDefinition",
    "ControlledVariableRecorder",
    "WorkerEnvironmentProbe",
]

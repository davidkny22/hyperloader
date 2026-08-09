"""Typed configuration for execution choices that do not belong to torch's surface."""

from .automatic import AUTO, Auto, AutoBytes, AutoFloat, AutoInt
from .control import CeilingConfig, ControlConfig
from .determinism import DeterminismConfig
from .distributed import DistributedConfig
from .executor import ExecutorConfig
from .factors import FactorConfig
from .io import IOConfig
from .memory import MemoryConfig
from .root import HyperConfig
from .scheduler import SchedulerConfig
from .telemetry import TelemetryConfig

_PUBLIC_CLASSES = (
    CeilingConfig,
    ControlConfig,
    DeterminismConfig,
    DistributedConfig,
    ExecutorConfig,
    FactorConfig,
    HyperConfig,
    IOConfig,
    MemoryConfig,
    SchedulerConfig,
    TelemetryConfig,
)
for _class in _PUBLIC_CLASSES:
    _class.__module__ = __name__
del _class

__all__ = [
    "AUTO",
    "Auto",
    "AutoBytes",
    "AutoFloat",
    "AutoInt",
    "CeilingConfig",
    "ControlConfig",
    "DeterminismConfig",
    "DistributedConfig",
    "ExecutorConfig",
    "FactorConfig",
    "HyperConfig",
    "IOConfig",
    "MemoryConfig",
    "SchedulerConfig",
    "TelemetryConfig",
]

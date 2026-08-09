"""Root hyperloader configuration."""

from dataclasses import dataclass, field

from .control import ControlConfig
from .determinism import DeterminismConfig
from .distributed import DistributedConfig
from .executor import ExecutorConfig
from .factors import FactorConfig
from .io import IOConfig
from .memory import MemoryConfig
from .scheduler import SchedulerConfig
from .telemetry import TelemetryConfig


@dataclass(frozen=True, slots=True)
class HyperConfig:
    """Collect every hyperloader-specific execution and contract setting."""

    seed: int | None = None
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    io: IOConfig = field(default_factory=IOConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    determinism: DeterminismConfig = field(default_factory=DeterminismConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    factors: FactorConfig = field(default_factory=FactorConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

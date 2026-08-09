"""Resource ceiling and controller cadence configuration."""

from dataclasses import dataclass, field

from .automatic import AUTO, AutoFloat, AutoInt, _require_nonnegative_int


@dataclass(frozen=True, slots=True)
class CeilingConfig:
    """Configure user-imposed controller ceilings."""

    cpu_cores: AutoInt = AUTO
    bandwidth: AutoFloat = AUTO

    def __post_init__(self) -> None:
        _require_nonnegative_int("control.ceilings.cpu_cores", self.cpu_cores)
        if self.bandwidth is not AUTO and self.bandwidth < 0:
            raise ValueError("control.ceilings.bandwidth must be auto or nonnegative")


@dataclass(frozen=True, slots=True)
class ControlConfig:
    """Configure controller cadence and resource ceilings."""

    cadence: AutoFloat = AUTO
    ceilings: CeilingConfig = field(default_factory=CeilingConfig)

    def __post_init__(self) -> None:
        if self.cadence is not AUTO and self.cadence <= 0:
            raise ValueError("control.cadence must be auto or positive")

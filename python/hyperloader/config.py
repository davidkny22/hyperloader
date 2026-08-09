"""Typed configuration for execution choices that do not belong to torch's surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal


class _Auto:
    """Represent a value derived from measured quantities at plan time."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "auto"


AUTO: Final = _Auto()
"""The singleton used for values derived at plan time."""

Auto = _Auto
AutoInt = int | _Auto
AutoFloat = float | _Auto
AutoBytes = int | _Auto


def _require_nonnegative_int(name: str, value: AutoInt) -> None:
    if value is not AUTO and (isinstance(value, bool) or value < 0):
        raise ValueError(f"{name} must be auto or a nonnegative integer")


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
    """Configure process capacity, placement, and worker-death behavior."""

    process_ceiling: AutoInt = AUTO
    pin_policy: Literal["auto", "efficiency-first", "none"] = "auto"
    on_worker_death: Literal["close", "restart"] = "close"

    def __post_init__(self) -> None:
        _require_nonnegative_int("executor.process_ceiling", self.process_ceiling)


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Configure frontier sizing, dispatch, and persisted cost profiles."""

    frontier_depth: AutoInt = AUTO
    frontier_budget: AutoBytes = AUTO
    early_dispatch: bool = True
    profile_cache: object = AUTO

    def __post_init__(self) -> None:
        _require_nonnegative_int("scheduler.frontier_depth", self.frontier_depth)
        _require_nonnegative_int("scheduler.frontier_budget", self.frontier_budget)


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Configure arena storage, shape information, delivery, and growth."""

    arena_backend: Literal["auto", "shm", "winmap", "unified"] = "auto"
    batch_shape: object = AUTO
    delivery_memory: Literal["auto", "host", "pinned", "device"] = "auto"
    growth: Literal["safe", "strict-error"] = "safe"


@dataclass(frozen=True, slots=True)
class IOConfig:
    """Configure platform I/O selection and direct-I/O policy."""

    backend: Literal["auto", "uring", "iocp", "pread"] = "auto"
    direct: Literal["auto", "on", "off"] = "auto"


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


@dataclass(frozen=True, slots=True)
class DeterminismConfig:
    """Configure result-observable determinism contracts."""

    exact_count: bool = False
    fingerprint: Literal["content", "strict"] = "content"
    decoder_pins: object = AUTO
    seeded_libs: _Auto | tuple[Literal["torch", "random", "numpy"], ...] = AUTO
    compat_resume: Literal["off", "on"] = "off"

    def __post_init__(self) -> None:
        if self.seeded_libs is AUTO:
            return
        unknown = set(self.seeded_libs) - {"torch", "random", "numpy"}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"determinism.seeded_libs contains unknown libraries: {names}")
        if len(set(self.seeded_libs)) != len(self.seeded_libs):
            raise ValueError("determinism.seeded_libs must not contain duplicates")


@dataclass(frozen=True, slots=True)
class TelemetryConfig:
    """Configure native instruments and benchmark-mode discipline."""

    enabled: bool = True
    benchmark_mode: bool = False


@dataclass(frozen=True, slots=True)
class FactorConfig:
    """Expose every named sizing and controller factor."""

    f_safety: float = 1.5
    f_mem: float = 0.15
    f_meta: float = 2.0
    f_q: float = 1.5
    f_cache: float = 2.0
    f_prof: float = 0.01
    f_stall: float = 0.001
    f_var: float = 8.0
    f_snap: int | Literal["off"] = 1
    f_snap_bytes: int = 4 * 1024 * 1024
    f_cad_s: float = 2.0
    f_cad_b: int = 20
    f_attach: float = 10.0
    alpha: float = 0.3
    d_min: AutoInt = AUTO
    b_buf: int = 2
    step_clip: int = 1
    hysteresis: int = 3
    growth_mult: int = 2

    def __post_init__(self) -> None:
        for name in (
            "f_safety",
            "f_mem",
            "f_meta",
            "f_q",
            "f_cache",
            "f_prof",
            "f_stall",
            "f_var",
            "f_snap_bytes",
            "f_cad_s",
            "f_cad_b",
            "f_attach",
            "alpha",
            "b_buf",
            "step_clip",
            "hysteresis",
            "growth_mult",
        ):
            _require_positive(f"factors.{name}", getattr(self, name))
        if self.f_snap != "off":
            _require_positive("factors.f_snap", self.f_snap)
        _require_nonnegative_int("factors.d_min", self.d_min)


@dataclass(frozen=True, slots=True)
class DistributedConfig:
    """Configure an explicit rank and world size when discovery is unavailable."""

    rank: AutoInt = AUTO
    world_size: AutoInt = AUTO

    def __post_init__(self) -> None:
        _require_nonnegative_int("distributed.rank", self.rank)
        _require_nonnegative_int("distributed.world_size", self.world_size)
        if self.world_size is not AUTO and self.world_size == 0:
            raise ValueError("distributed.world_size must be auto or positive")
        if (
            self.rank is not AUTO
            and self.world_size is not AUTO
            and self.rank >= self.world_size
        ):
            raise ValueError("distributed.rank must be smaller than distributed.world_size")


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

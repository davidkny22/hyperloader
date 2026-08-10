"""Named sizing and controller factors."""

from dataclasses import dataclass
from typing import Literal

from .automatic import AUTO, AutoInt, _require_nonnegative_int


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


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
        if self.alpha > 1:
            raise ValueError("factors.alpha must not exceed one")
        _require_nonnegative_int("factors.d_min", self.d_min)

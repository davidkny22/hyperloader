"""Result-observable determinism configuration."""

from dataclasses import dataclass
from typing import Literal

from .automatic import AUTO, _Auto


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
            raise ValueError(
                f"determinism.seeded_libs contains unknown libraries: {names}"
            )
        if len(set(self.seeded_libs)) != len(self.seeded_libs):
            raise ValueError("determinism.seeded_libs must not contain duplicates")

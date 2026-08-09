"""Arena and delivery-memory configuration."""

from dataclasses import dataclass
from typing import Literal

from .automatic import AUTO


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Configure arena storage, shape information, delivery, and growth."""

    arena_backend: Literal["auto", "shm", "winmap", "unified"] = "auto"
    batch_shape: object = AUTO
    delivery_memory: Literal["auto", "host", "pinned", "device"] = "auto"
    growth: Literal["safe", "strict-error"] = "safe"

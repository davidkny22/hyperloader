"""Per-class accounting for loader-attributable batch bytes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ByteLedger:
    """Accumulate explicit pinned-stage and final-write byte events."""

    source_class: str
    delivery: str
    _produced_batches: int = field(default=0, init=False, repr=False)
    _produced_samples: int = field(default=0, init=False, repr=False)
    _payload_bytes: int = field(default=0, init=False, repr=False)
    _pinned_stage_bytes: int = field(default=0, init=False, repr=False)
    _arena_write_bytes: int = field(default=0, init=False, repr=False)

    def record(
        self,
        value: Any,
        samples: int,
        *,
        pinned_stage_bytes: int = 0,
        arena_write_bytes: int = 0,
    ) -> None:
        """Record one produced batch from events at its ownership boundaries."""
        if samples < 0 or pinned_stage_bytes < 0 or arena_write_bytes < 0:
            raise ValueError("byte-accounting values must be nonnegative")
        self._produced_batches += 1
        self._produced_samples += samples
        self._payload_bytes += payload_bytes(value)
        self._pinned_stage_bytes += pinned_stage_bytes
        self._arena_write_bytes += arena_write_bytes

    def report(self) -> dict[str, object]:
        """Return actual and irreducible traffic on the same counting basis."""
        actual = self._pinned_stage_bytes + self._arena_write_bytes
        irreducible = actual
        samples = self._produced_samples
        divisor = samples if samples else 1
        return {
            "actual_bytes": actual,
            "actual_bytes_per_sample": actual / divisor if samples else 0.0,
            "arena_write_bytes": self._arena_write_bytes,
            "bytes_beyond_irreducible": actual - irreducible,
            "bytes_beyond_irreducible_per_sample": 0.0,
            "delivery": self.delivery,
            "irreducible_bytes": irreducible,
            "irreducible_bytes_per_sample": (irreducible / divisor if samples else 0.0),
            "payload_bytes": self._payload_bytes,
            "pinned_stage_bytes": self._pinned_stage_bytes,
            "produced_batches": self._produced_batches,
            "produced_samples": samples,
            "source_class": self.source_class,
        }


def payload_bytes(value: Any) -> int:
    """Count payload bytes without materializing or traversing tensor elements."""
    if _is_torch_tensor(value):
        return int(value.numel() * value.element_size())
    if _is_numpy_array(value):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(payload_bytes(item) for item in value.values())
    if isinstance(value, (str, bytes, bytearray)):
        return len(value.encode("utf-8") if isinstance(value, str) else value)
    if isinstance(value, Sequence):
        return sum(payload_bytes(item) for item in value)
    if isinstance(value, (bool, int, float)):
        return 8
    return 0


def _is_torch_tensor(value: Any) -> bool:
    value_type = type(value)
    return value_type.__module__ == "torch" and value_type.__name__ == "Tensor"


def _is_numpy_array(value: Any) -> bool:
    try:
        import numpy as np
    except ImportError:
        return False
    return isinstance(value, np.ndarray)

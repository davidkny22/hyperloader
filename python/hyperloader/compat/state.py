"""Public state representation for zero-worker torch compatibility."""

from __future__ import annotations

from dataclasses import dataclass

from hyperloader.fingerprint import ContractFingerprint

from .rng import validate_globals


@dataclass(frozen=True, slots=True)
class CompatZeroCheckpoint:
    """Carry sampler progress and ambient RNG states for exact continuation."""

    delivered_batches: int
    iterator_globals: dict[str, bytes]
    current_globals: dict[str, bytes]
    iterator_generator: bytes | None
    current_generator: bytes | None
    fingerprint: ContractFingerprint

    def to_dict(self) -> dict[str, object]:
        """Return the public zero-worker checkpoint representation."""
        return {
            "kind": "compat-zero",
            "sampler_position": self.delivered_batches,
            "delivered_cursor": self.delivered_batches,
            "iterator_globals": dict(self.iterator_globals),
            "current_globals": dict(self.current_globals),
            "iterator_generator": self.iterator_generator,
            "current_generator": self.current_generator,
            "fingerprint": self.fingerprint.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: object) -> CompatZeroCheckpoint:
        """Validate a public zero-worker checkpoint representation."""
        if not isinstance(payload, dict):
            raise TypeError("loader state must be a dictionary")
        if payload.get("kind") != "compat-zero":
            raise ValueError("compat zero-worker state requires kind='compat-zero'")
        sampler_position = _nonnegative_integer(payload, "sampler_position")
        delivered_cursor = _nonnegative_integer(payload, "delivered_cursor")
        if sampler_position != delivered_cursor:
            raise ValueError("compat sampler position must match delivered cursor")
        fingerprint = payload.get("fingerprint")
        if not isinstance(fingerprint, dict):
            raise TypeError("loader state fingerprint must be a dictionary")
        iterator_generator = _optional_bytes(payload, "iterator_generator")
        current_generator = _optional_bytes(payload, "current_generator")
        if (iterator_generator is None) != (current_generator is None):
            raise ValueError("compat generator states must appear together")
        return cls(
            delivered_batches=delivered_cursor,
            iterator_globals=validate_globals(payload.get("iterator_globals")),
            current_globals=validate_globals(payload.get("current_globals")),
            iterator_generator=iterator_generator,
            current_generator=current_generator,
            fingerprint=ContractFingerprint.from_dict(fingerprint),
        )


def _nonnegative_integer(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"compat loader state {key} must be an integer")
    if value < 0:
        raise ValueError(f"compat loader state {key} must be nonnegative")
    return value


def _optional_bytes(payload: dict[str, object], key: str) -> bytes | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, bytes):
        raise TypeError(f"compat loader state {key} must be bytes or None")
    return value

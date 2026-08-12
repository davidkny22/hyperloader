"""Checkpoint representation for torch-compatible worker lanes."""

from __future__ import annotations

from dataclasses import dataclass

from hyperloader.fingerprint import ContractFingerprint


@dataclass(frozen=True, slots=True)
class CompatMultiCheckpoint:
    """Carry a strict delivered prefix and per-lane restore snapshots."""

    delivered_batches: int
    worker_count: int
    assignment_phase: int
    reused_base_seed: bool
    iterator_generator: bytes
    current_generator: bytes
    lane_states: dict[int, bytes]
    lane_seeds: dict[int, int]
    fingerprint: ContractFingerprint

    def to_dict(self) -> dict[str, object]:
        """Return the public multi-worker checkpoint representation."""
        return {
            "kind": "compat-multi",
            "sampler_position": self.delivered_batches,
            "delivered_cursor": self.delivered_batches,
            "num_workers": self.worker_count,
            "assignment_phase": self.assignment_phase,
            "reused_base_seed": self.reused_base_seed,
            "iterator_generator": self.iterator_generator,
            "current_generator": self.current_generator,
            "lane_states": dict(self.lane_states),
            "lane_seeds": dict(self.lane_seeds),
            "fingerprint": self.fingerprint.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: object) -> CompatMultiCheckpoint:
        """Validate a public multi-worker checkpoint representation."""
        if not isinstance(payload, dict):
            raise TypeError("loader state must be a dictionary")
        if payload.get("kind") != "compat-multi":
            raise ValueError("compat worker state requires kind='compat-multi'")
        delivered = _nonnegative_integer(payload, "delivered_cursor")
        if _nonnegative_integer(payload, "sampler_position") != delivered:
            raise ValueError("compat sampler position must match delivered cursor")
        workers = _positive_integer(payload, "num_workers")
        phase = _nonnegative_integer(payload, "assignment_phase")
        if phase != delivered % workers:
            raise ValueError("compat assignment phase does not match delivered cursor")
        reused_base_seed = payload.get("reused_base_seed")
        if not isinstance(reused_base_seed, bool):
            raise TypeError("compat reused_base_seed must be a boolean")
        lane_states = payload.get("lane_states")
        if not isinstance(lane_states, dict):
            raise TypeError("compat lane states must be a dictionary")
        validated = {}
        for lane, state in lane_states.items():
            if isinstance(lane, bool) or not isinstance(lane, int):
                raise TypeError("compat lane identifiers must be integers")
            if lane < 0 or lane >= workers:
                raise ValueError("compat lane identifier is outside num_workers")
            if not isinstance(state, bytes):
                raise TypeError("compat lane RNG state must be bytes")
            validated[lane] = state
        lane_seeds = _lane_seeds(payload.get("lane_seeds"), workers)
        if set(lane_seeds) != set(validated):
            raise ValueError("compat lane states and seeds must name the same lanes")
        fingerprint = payload.get("fingerprint")
        if not isinstance(fingerprint, dict):
            raise TypeError("loader state fingerprint must be a dictionary")
        return cls(
            delivered_batches=delivered,
            worker_count=workers,
            assignment_phase=phase,
            reused_base_seed=reused_base_seed,
            iterator_generator=_bytes(payload, "iterator_generator"),
            current_generator=_bytes(payload, "current_generator"),
            lane_states=validated,
            lane_seeds=lane_seeds,
            fingerprint=ContractFingerprint.from_dict(fingerprint),
        )


def _lane_seeds(payload: object, workers: int) -> dict[int, int]:
    if not isinstance(payload, dict):
        raise TypeError("compat lane seeds must be a dictionary")
    validated = {}
    for lane, seed in payload.items():
        if isinstance(lane, bool) or not isinstance(lane, int):
            raise TypeError("compat lane seed identifiers must be integers")
        if lane < 0 or lane >= workers:
            raise ValueError("compat lane seed identifier is outside num_workers")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("compat lane seed must be an integer")
        validated[lane] = seed
    return validated


def _nonnegative_integer(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"compat loader state {key} must be an integer")
    if value < 0:
        raise ValueError(f"compat loader state {key} must be nonnegative")
    return value


def _positive_integer(payload: dict[str, object], key: str) -> int:
    value = _nonnegative_integer(payload, key)
    if value == 0:
        raise ValueError(f"compat loader state {key} must be positive")
    return value


def _bytes(payload: dict[str, object], key: str) -> bytes:
    value = payload.get(key)
    if not isinstance(value, bytes):
        raise TypeError(f"compat loader state {key} must be bytes")
    return value

"""Public coordinate state for exact iterable source continuation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hyperloader.fingerprint import ContractFingerprint, require_fingerprint_match

from .factory import logical_lane_count

MAX_U64 = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class LaneCheckpoint:
    """Carry one lane's delivered cursor and selected source snapshot."""

    lane: int
    delivered_arrival: int
    stateful: bool
    snapshot_arrival: int | None
    snapshot: bytes | None

    def to_dict(self) -> dict[str, object]:
        """Return the public representation of one lane checkpoint."""
        return {
            "lane": self.lane,
            "delivered_arrival": self.delivered_arrival,
            "stateful": self.stateful,
            "snapshot_arrival": self.snapshot_arrival,
            "snapshot": self.snapshot,
        }

    @classmethod
    def from_dict(cls, payload: object) -> LaneCheckpoint:
        """Validate one public lane checkpoint."""
        if not isinstance(payload, dict):
            raise TypeError("iterable lane state must be a dictionary")
        stateful = payload.get("stateful")
        if not isinstance(stateful, bool):
            raise TypeError("iterable lane stateful must be a boolean")
        snapshot = payload.get("snapshot")
        if snapshot is not None and not isinstance(snapshot, bytes):
            raise TypeError("iterable lane snapshot must be bytes or None")
        raw_arrival = payload.get("snapshot_arrival")
        snapshot_arrival = (
            None
            if raw_arrival is None
            else _integer(payload, "snapshot_arrival")
        )
        if (snapshot is None) != (snapshot_arrival is None):
            raise ValueError(
                "iterable lane snapshot and snapshot_arrival must appear together"
            )
        if not stateful and snapshot is not None:
            raise ValueError("plain iterable lane cannot carry a source snapshot")
        delivered = _integer(payload, "delivered_arrival")
        if snapshot_arrival is not None and snapshot_arrival > delivered:
            raise ValueError("iterable lane snapshot exceeds its delivered arrival")
        return cls(
            lane=_integer(payload, "lane"),
            delivered_arrival=delivered,
            stateful=stateful,
            snapshot_arrival=snapshot_arrival,
            snapshot=snapshot,
        )


@dataclass(frozen=True, slots=True)
class IterableCheckpoint:
    """Carry exact per-lane iterable continuation state."""

    root_seed: int
    epoch: int
    world_size: int
    lane_count: int
    lane_order: tuple[int, ...]
    lanes: tuple[LaneCheckpoint, ...]
    fingerprint: ContractFingerprint

    def to_dict(self) -> dict[str, object]:
        """Return the public iterable checkpoint representation."""
        return {
            "kind": "iterable",
            "root_seed": self.root_seed,
            "epoch": self.epoch,
            "world_size": self.world_size,
            "lane_count": self.lane_count,
            "lane_order": list(self.lane_order),
            "lanes": [lane.to_dict() for lane in self.lanes],
            "fingerprint": self.fingerprint.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: object) -> IterableCheckpoint:
        """Validate a public iterable checkpoint representation."""
        if not isinstance(payload, dict):
            raise TypeError("loader state must be a dictionary")
        if payload.get("kind") != "iterable":
            raise ValueError("iterable loader state requires kind='iterable'")
        fingerprint_payload = payload.get("fingerprint")
        if not isinstance(fingerprint_payload, dict):
            raise TypeError("loader state fingerprint must be a dictionary")
        lane_count = _integer(payload, "lane_count")
        raw_lanes = payload.get("lanes")
        if not isinstance(raw_lanes, list):
            raise TypeError("iterable loader state lanes must be a list")
        lanes = tuple(LaneCheckpoint.from_dict(item) for item in raw_lanes)
        lane_ids = tuple(lane.lane for lane in lanes)
        if lanes and tuple(sorted(lane_ids)) != tuple(range(lane_count)):
            raise ValueError(
                "iterable loader state lanes must contain every logical lane once"
            )
        raw_order = payload.get("lane_order")
        if not isinstance(raw_order, list):
            raise TypeError("iterable loader state lane_order must be a list")
        lane_order = tuple(
            _sequence_integer(value, "lane_order") for value in raw_order
        )
        if len(lane_order) != len(set(lane_order)) or any(
            lane >= lane_count for lane in lane_order
        ):
            raise ValueError(
                "iterable loader state lane_order must contain unique valid lanes"
            )
        return cls(
            root_seed=_integer(payload, "root_seed", maximum=MAX_U64),
            epoch=_integer(payload, "epoch"),
            world_size=_integer(payload, "world_size"),
            lane_count=lane_count,
            lane_order=lane_order,
            lanes=lanes,
            fingerprint=ContractFingerprint.from_dict(fingerprint_payload),
        )


def capture_iterable_state(loader: Any) -> dict[str, object]:
    """Capture source states already resident beside the active lane set."""
    active = (
        None if loader._active_iterator_ref is None else loader._active_iterator_ref()
    )
    if active is None and loader._resume_iterable_state is not None:
        return loader._resume_iterable_state.to_dict()
    if active is not None and not active.complete:
        return active.capture_checkpoint().to_dict()
    lanes = logical_lane_count(loader)
    return IterableCheckpoint(
        root_seed=loader.root_seed,
        epoch=loader._epoch_state.current,
        world_size=loader._distributed_topology.world_size,
        lane_count=lanes,
        lane_order=tuple(range(lanes)),
        lanes=(),
        fingerprint=loader._fingerprint,
    ).to_dict()


def restore_iterable_state(loader: Any, payload: dict[str, object]) -> None:
    """Validate and install one iterable checkpoint for the next iterator."""
    state = IterableCheckpoint.from_dict(payload)
    require_fingerprint_match(state.fingerprint, loader._fingerprint)
    lanes = logical_lane_count(loader)
    if state.world_size != loader._distributed_topology.world_size:
        raise ValueError("iterable resume requires the same world_size")
    if state.lane_count != lanes:
        raise ValueError("iterable resume requires the same logical lane count")
    loader.close()
    loader.root_seed = state.root_seed
    loader._epoch_state.restore(state.epoch)
    loader._resume_iterable_state = state


def _integer(
    payload: dict[str, object], key: str, *, maximum: int | None = None
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"iterable loader state {key} must be an integer")
    if value < 0:
        raise ValueError(f"iterable loader state {key} must be nonnegative")
    if maximum is not None and value > maximum:
        raise ValueError(f"iterable loader state {key} exceeds its supported range")
    return value


def _sequence_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"iterable loader state {name} entries must be integers")
    if value < 0:
        raise ValueError(f"iterable loader state {name} entries must be nonnegative")
    return value

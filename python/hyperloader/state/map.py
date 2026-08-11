"""Bytes-sized map-style coordinate state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..fingerprint import ContractFingerprint, require_fingerprint_match
from .delivery import decode_delivered_bitmap

MAX_U64 = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class MapCoordinateState:
    """Carry one delivered-prefix coordinate and its result contract."""

    root_seed: int
    epoch: int
    cursor: int
    global_batch: int
    sampler_checksum: int
    delivered_bitmap: bytes
    fingerprint: ContractFingerprint
    batch_shape: Any

    def to_dict(self) -> dict[str, object]:
        """Return the public checkpoint representation."""
        return {
            "root_seed": self.root_seed,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "B_g": self.global_batch,
            "sampler_checksum": self.sampler_checksum,
            "delivered_bitmap": self.delivered_bitmap,
            "fingerprint": self.fingerprint.to_dict(),
            "batch_shape": self.batch_shape,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> MapCoordinateState:
        """Validate a public checkpoint representation."""
        if not isinstance(payload, dict):
            raise TypeError("loader state must be a dictionary")
        fingerprint_payload = payload.get("fingerprint")
        if not isinstance(fingerprint_payload, dict):
            raise TypeError("loader state fingerprint must be a dictionary")
        return cls(
            root_seed=_integer(payload, "root_seed", maximum=MAX_U64),
            epoch=_integer(payload, "epoch"),
            cursor=_integer(payload, "cursor"),
            global_batch=_integer(payload, "B_g"),
            sampler_checksum=_integer(payload, "sampler_checksum", maximum=MAX_U64),
            delivered_bitmap=_byte_string(payload, "delivered_bitmap"),
            fingerprint=ContractFingerprint.from_dict(fingerprint_payload),
            batch_shape=payload.get("batch_shape"),
        )


def capture_map_state(loader: Any) -> dict[str, object]:
    """Capture the delivered prefix without serializing speculative work."""
    active = (
        None if loader._active_iterator_ref is None else loader._active_iterator_ref()
    )
    if active is None and loader._resume_cursor_batches:
        epoch = loader._epoch_state.current
        cursor = loader._resume_cursor_batches
        checksum = loader._resume_sampler_checksum
        delivered_bitmap = loader._resume_delivered_bitmap
    elif active is None or active.complete:
        epoch = loader._epoch_state.current
        cursor = 0
        checksum = 0
        delivered_bitmap = b""
    else:
        epoch = active.coordinate_epoch
        cursor = active.delivered_batches
        checksum = (
            active.sampler_checksum
            if loader.sampler is not None or loader.batch_sampler is not None
            else 0
        )
        delivered_bitmap = bytes(getattr(active, "delivered_bitmap", b""))
    fingerprint = loader._fingerprint
    return MapCoordinateState(
        root_seed=loader.root_seed,
        epoch=epoch,
        cursor=cursor,
        global_batch=int(_fingerprint_value(fingerprint, "placement.B_g")),
        sampler_checksum=checksum,
        delivered_bitmap=delivered_bitmap,
        fingerprint=fingerprint,
        batch_shape=_fingerprint_value(fingerprint, "batch_shape"),
    ).to_dict()


def restore_map_state(loader: Any, payload: dict[str, object]) -> None:
    """Validate and install one coordinate for the next iterator."""
    state = MapCoordinateState.from_dict(payload)
    require_fingerprint_match(state.fingerprint, loader._fingerprint)
    current_batch = int(_fingerprint_value(loader._fingerprint, "placement.B_g"))
    if state.global_batch != current_batch:
        raise ValueError(
            "loader state B_g does not match the current placement.B_g fingerprint"
        )
    current_shape = _fingerprint_value(loader._fingerprint, "batch_shape")
    if state.batch_shape != current_shape:
        raise ValueError(
            "loader state batch_shape does not match the current fingerprint"
        )
    if (
        loader.sampler is None
        and loader.batch_sampler is None
        and state.sampler_checksum != 0
    ):
        raise ValueError("native sampler state requires sampler_checksum=0")
    if loader.delivery == "in-order" and state.delivered_bitmap:
        raise ValueError("in-order loader state requires an empty delivered_bitmap")
    if loader.delivery == "on-completion" and state.delivered_bitmap:
        width = loader.batch_size or 1
        total_batches = (loader._plan.length + width - 1) // width
        decode_delivered_bitmap(state.cursor, state.delivered_bitmap, total_batches)
    loader.close()
    loader.root_seed = state.root_seed
    loader._epoch_state.restore(state.epoch)
    loader._resume_cursor_batches = state.cursor
    loader._resume_sampler_checksum = state.sampler_checksum
    loader._resume_delivered_bitmap = state.delivered_bitmap


def _fingerprint_value(fingerprint: ContractFingerprint, path: str) -> Any:
    for element in fingerprint.elements:
        if element.path == path:
            return element.value
    raise RuntimeError(f"contract fingerprint is missing {path}")


def _integer(
    payload: dict[str, object], key: str, *, maximum: int | None = None
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"loader state {key} must be an integer")
    if value < 0:
        raise ValueError(f"loader state {key} must be nonnegative")
    if maximum is not None and value > maximum:
        raise ValueError(f"loader state {key} exceeds its supported range")
    return value


def _byte_string(payload: dict[str, object], key: str) -> bytes:
    value = payload.get(key, b"")
    if not isinstance(value, bytes):
        raise TypeError(f"loader state {key} must be bytes")
    return value

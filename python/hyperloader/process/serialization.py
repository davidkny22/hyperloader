"""Process-safe dataset and successful-result serialization."""

from __future__ import annotations

import pickle
from multiprocessing.reduction import ForkingPickler
from typing import Any

MAGIC = b"HLRES\x00"
PICKLE_VALUE = 0
TENSOR_BASE = 1
TENSOR_VIEW = 2


def encode_multiprocessing(value: Any) -> bytes:
    """Encode a process argument with platform storage reducers."""
    return bytes(ForkingPickler.dumps(value, protocol=5))


class ResultEncoder:
    """Share each top-level tensor storage once, then send view coordinates."""

    def __init__(self) -> None:
        self._storage_ids: dict[int, int] = {}

    def encode(self, value: Any) -> bytes:
        """Encode one successful worker value."""
        if _shareable_tensor(value):
            storage_key = value.untyped_storage()._cdata
            storage_id = self._storage_ids.get(storage_key)
            if storage_id is None:
                storage_id = len(self._storage_ids)
                self._storage_ids[storage_key] = storage_id
                return _envelope(
                    TENSOR_BASE,
                    ForkingPickler.dumps((storage_id, value), protocol=5),
                )
            coordinates = (
                storage_id,
                tuple(value.size()),
                tuple(value.stride()),
                value.storage_offset(),
                value.requires_grad,
            )
            return _envelope(TENSOR_VIEW, pickle.dumps(coordinates, protocol=5))
        return _envelope(PICKLE_VALUE, ForkingPickler.dumps(value, protocol=5))


class ResultDecoder:
    """Reconstruct successful values and retain shared tensor storage owners."""

    def __init__(self) -> None:
        self._tensor_bases: dict[tuple[int, int], Any] = {}

    def decode(self, payload: bytes, worker: int) -> Any:
        """Decode one successful value from a named worker."""
        if not payload.startswith(MAGIC) or len(payload) == len(MAGIC):
            raise RuntimeError("Worker result envelope is invalid.")
        kind = payload[len(MAGIC)]
        body = payload[len(MAGIC) + 1 :]
        if kind == PICKLE_VALUE:
            return pickle.loads(body)
        if kind == TENSOR_BASE:
            storage_id, value = pickle.loads(body)
            self._tensor_bases[(worker, storage_id)] = value
            return value
        if kind != TENSOR_VIEW:
            raise RuntimeError("Worker result envelope kind is invalid.")
        storage_id, size, stride, offset, requires_grad = pickle.loads(body)
        try:
            base = self._tensor_bases[(worker, storage_id)]
        except KeyError as error:
            raise RuntimeError("Worker tensor storage reference is unknown.") from error
        value = base.new_empty(0)
        value.set_(base.untyped_storage(), offset, size, stride)
        return value.requires_grad_(requires_grad)


def _shareable_tensor(value: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return (
        isinstance(value, torch.Tensor)
        and value.device.type == "cpu"
        and value.layout == torch.strided
        and not value.is_conj()
        and not value.is_neg()
    )


def _envelope(kind: int, body: bytes | memoryview) -> bytes:
    return MAGIC + bytes((kind,)) + bytes(body)

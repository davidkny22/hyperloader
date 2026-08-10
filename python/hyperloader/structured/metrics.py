"""Payload accounting for batch-native structured delivery."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


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

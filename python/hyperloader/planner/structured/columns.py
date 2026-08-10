"""Default-collation-compatible conversion of batched column values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def collate_columns(value: Any) -> Any:
    """Convert one batch-native column tree without per-row reconstruction."""
    if _is_torch_tensor(value):
        return value
    if _is_numpy_array(value):
        if value.dtype.kind in {"O", "S", "U"}:
            return _collate_list(value.tolist())
        import torch

        return torch.from_numpy(value)
    if isinstance(value, Mapping):
        items = [(key, collate_columns(item)) for key, item in value.items()]
        try:
            return type(value)(items)
        except TypeError:
            clone = value.copy()
            clone.update(items)
            return clone
    if isinstance(value, tuple):
        return tuple(collate_columns(item) for item in value)
    if isinstance(value, list):
        return _collate_list(value)
    return value


def _collate_list(values: list[Any]) -> Any:
    if not values or all(isinstance(item, (str, bytes)) for item in values):
        return values
    import torch

    if all(isinstance(item, bool) for item in values):
        return torch.tensor(values, dtype=torch.bool)
    if all(isinstance(item, int) and not isinstance(item, bool) for item in values):
        return torch.tensor(values, dtype=torch.int64)
    if all(isinstance(item, float) for item in values):
        return torch.tensor(values, dtype=torch.float64)
    if all(isinstance(item, Mapping) for item in values):
        first = values[0]
        items = [
            (key, collate_columns([item[key] for item in values])) for key in first
        ]
        try:
            return type(first)(items)
        except TypeError:
            clone = first.copy()
            clone.update(items)
            return clone
    if all(isinstance(item, (list, tuple)) for item in values):
        length = len(values[0])
        if all(len(item) == length for item in values):
            return [
                collate_columns([item[index] for item in values])
                for index in range(length)
            ]
    return values


def _is_torch_tensor(value: Any) -> bool:
    value_type = type(value)
    return value_type.__module__ == "torch" and value_type.__name__ == "Tensor"


def _is_numpy_array(value: Any) -> bool:
    try:
        import numpy as np
    except ImportError:
        return False
    return isinstance(value, np.ndarray)

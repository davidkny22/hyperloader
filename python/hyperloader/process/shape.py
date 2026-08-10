"""Probe-derived nested batch-shape descriptions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def batch_shape(value: Any, batch_size: int | None) -> dict[str, object]:
    """Describe the public value or its default-collated batch without copying it."""
    if _is_torch_tensor(value):
        shape = [int(size) for size in value.shape]
        if batch_size is not None:
            shape.insert(0, batch_size)
        return {"dtype": str(value.dtype), "kind": "tensor", "shape": shape}
    if _is_numpy_array(value):
        shape = [int(size) for size in value.shape]
        if batch_size is None:
            return {"dtype": str(value.dtype), "kind": "ndarray", "shape": shape}
        import numpy as np
        import torch

        dtype = str(torch.from_numpy(np.empty(0, dtype=value.dtype)).dtype)
        return {"dtype": dtype, "kind": "tensor", "shape": [batch_size, *shape]}
    if _is_numpy_scalar(value):
        if batch_size is None:
            return {"dtype": str(value.dtype), "kind": "numpy-scalar", "shape": []}
        import torch

        return {
            "dtype": str(torch.as_tensor(value).dtype),
            "kind": "tensor",
            "shape": [batch_size],
        }
    if isinstance(value, Mapping):
        return {
            "items": [
                {"key": _stable_key(key), "value": batch_shape(item, batch_size)}
                for key, item in value.items()
            ],
            "kind": "mapping",
            "type": _qualified_type(value),
        }
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return {
            "items": [batch_shape(item, batch_size) for item in value],
            "kind": "namedtuple",
            "type": _qualified_type(value),
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {
            "items": [batch_shape(item, batch_size) for item in value],
            "kind": "sequence",
            "type": _qualified_type(value),
        }
    if batch_size is not None and isinstance(value, (bool, int, float)):
        dtype = (
            "torch.bool"
            if isinstance(value, bool)
            else "torch.float64"
            if isinstance(value, float)
            else "torch.int64"
        )
        return {"dtype": dtype, "kind": "tensor", "shape": [batch_size]}
    if isinstance(value, (str, bytes)):
        return {
            "item_type": _qualified_type(value),
            "kind": "sequence" if batch_size is not None else "scalar",
            "length": batch_size,
        }
    return {
        "batch_size": batch_size,
        "kind": "opaque",
        "type": _qualified_type(value),
    }


def _stable_key(value: Any) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"type": _qualified_type(value)}


def _qualified_type(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _is_torch_tensor(value: Any) -> bool:
    value_type = type(value)
    return value_type.__module__ == "torch" and value_type.__name__ == "Tensor"


def _is_numpy_array(value: Any) -> bool:
    try:
        import numpy as np
    except ImportError:
        return False
    return isinstance(value, np.ndarray)


def _is_numpy_scalar(value: Any) -> bool:
    try:
        import numpy as np
    except ImportError:
        return False
    return isinstance(value, np.generic)

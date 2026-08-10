"""Array shape, dtype, and strict byte-digest elements."""

from __future__ import annotations

import hashlib
import pickle
from typing import Any

from .model import FingerprintElement


def array_elements(array: Any, mode: str, prefix: str) -> list[FingerprintElement]:
    """Describe an array without hashing its values unless strict mode requests it."""
    elements = [
        FingerprintElement(f"{prefix}.shape", [int(value) for value in array.shape]),
        FingerprintElement(f"{prefix}.dtype", str(array.dtype)),
    ]
    if mode == "strict":
        elements.append(
            FingerprintElement(f"{prefix}.content_sha256", _array_digest(array))
        )
    return elements


def _array_digest(array: Any) -> str:
    value_type = type(array)
    if value_type.__module__ == "torch" and value_type.__name__ == "Tensor":
        import torch

        value = array.detach().cpu().contiguous().reshape(-1)
        try:
            raw = value.view(torch.uint8).numpy().tobytes()
        except (TypeError, RuntimeError):
            raw = pickle.dumps(value, protocol=5)
    else:
        raw = array.tobytes(order="A")
    return hashlib.sha256(raw).hexdigest()

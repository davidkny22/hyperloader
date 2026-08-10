"""Worker-side default collation for one native batch command."""

from __future__ import annotations

from typing import Any

from hyperloader import _hyperloader

from .serialization import ResultEncoder


def supports_worker_batch(value: Any) -> bool:
    """Return whether one probe proves the exact NumPy batch transport contract."""
    try:
        import numpy as np
    except ImportError:
        return False
    return (
        type(value) is np.ndarray
        and value.flags.c_contiguous
        and not value.dtype.hasobject
    )


def encode_batch(values: list[Any], encoder: ResultEncoder) -> bytes:
    """Collate one default-collation batch and encode its storage once."""
    batch = _hyperloader._default_collate(values)
    return encoder.encode_uncached(batch)

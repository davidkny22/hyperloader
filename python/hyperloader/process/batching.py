"""Worker-side default collation for one native batch command."""

from __future__ import annotations

from typing import Any

from hyperloader import _hyperloader

from .serialization import ResultEncoder


def supports_worker_batch(value: Any) -> bool:
    """Return whether one probe selects worker-collated transport."""
    return False


def encode_batch(values: list[Any], encoder: ResultEncoder) -> bytes:
    """Collate one default-collation batch and encode its storage once."""
    batch = _hyperloader._default_collate(values)
    return encoder.encode_uncached(batch)

"""Canonical delivered-batch hash chaining for training evidence."""

from __future__ import annotations

import hashlib

EMPTY_HASH_CHAIN = "0" * 64


def advance_hash_chain(previous: str, batch_digest: str) -> str:
    """Bind one precomputed batch digest to the preceding chain value."""
    try:
        previous_bytes = bytes.fromhex(previous)
        batch_bytes = bytes.fromhex(batch_digest)
    except ValueError as error:
        raise ValueError("hash-chain inputs must be hexadecimal") from error
    if len(previous_bytes) != 32 or len(batch_bytes) != 32:
        raise ValueError("hash-chain inputs must each contain 32 bytes")
    return hashlib.sha256(previous_bytes + batch_bytes).hexdigest()

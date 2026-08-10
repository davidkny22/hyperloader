"""Immutable selected decoder identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class DecoderSelection:
    """Disclose one stage's result-observable decoder selection."""

    stage: str
    codec: str
    backend: str
    version: str
    platform: str
    source: str
    substituted: bool

    def to_dict(self) -> dict[str, object]:
        """Return a fresh fingerprint-ready mapping."""
        return asdict(self)

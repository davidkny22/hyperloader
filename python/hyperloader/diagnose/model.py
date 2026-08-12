"""Stable values returned by loader diagnosis."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DiagnosisReport:
    """Pair a human-readable diagnosis with its machine-readable record."""

    text: str
    record: dict[str, Any]

    def __str__(self) -> str:
        return self.text

    def to_dict(self) -> dict[str, Any]:
        """Return an independent machine-readable record."""
        return deepcopy(self.record)

"""Immutable ordered fingerprints and mismatch diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FingerprintElement:
    """Name one canonical input to a dataset or result contract."""

    path: str
    value: Any

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("fingerprint element path must not be empty")


@dataclass(frozen=True, slots=True)
class ContractFingerprint:
    """Carry ordered canonical elements and their SHA-256 digest."""

    elements: tuple[FingerprintElement, ...]

    def __post_init__(self) -> None:
        paths = [element.path for element in self.elements]
        if len(paths) != len(set(paths)):
            raise ValueError("fingerprint element paths must be unique")
        _canonical_bytes(self.elements)

    @property
    def digest(self) -> str:
        """Return the canonical SHA-256 digest."""
        return hashlib.sha256(_canonical_bytes(self.elements)).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return a state-ready representation with inspectable elements."""
        return {
            "digest": self.digest,
            "elements": [
                {"path": element.path, "value": element.value}
                for element in self.elements
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ContractFingerprint:
        """Validate a state representation and its claimed digest."""
        raw_elements = payload.get("elements")
        if not isinstance(raw_elements, list):
            raise ValueError("fingerprint elements must be a list")
        elements = []
        for raw in raw_elements:
            if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
                raise ValueError("fingerprint elements must contain string paths")
            elements.append(FingerprintElement(raw["path"], raw.get("value")))
        fingerprint = cls(tuple(elements))
        if payload.get("digest") != fingerprint.digest:
            raise ValueError("fingerprint digest does not match its elements")
        return fingerprint


def require_fingerprint_match(
    expected: ContractFingerprint, actual: ContractFingerprint
) -> None:
    """Raise with the first changed, missing, or added contract element."""
    expected_values = {element.path: element.value for element in expected.elements}
    actual_values = {element.path: element.value for element in actual.elements}
    ordered_paths = [element.path for element in expected.elements]
    ordered_paths.extend(
        element.path
        for element in actual.elements
        if element.path not in expected_values
    )
    for path in ordered_paths:
        old = expected_values.get(path, _MISSING)
        new = actual_values.get(path, _MISSING)
        if old != new:
            raise ValueError(
                "fingerprint mismatch at "
                f"{path}: expected {_render(old)}, found {_render(new)}"
            )


_MISSING = object()


def _canonical_bytes(elements: tuple[FingerprintElement, ...]) -> bytes:
    payload = [{"path": element.path, "value": element.value} for element in elements]
    try:
        document = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise TypeError("fingerprint values must be canonical JSON values") from error
    return document.encode("utf-8")


def _render(value: object) -> str:
    if value is _MISSING:
        return "<missing>"
    return json.dumps(value, ensure_ascii=False, sort_keys=True)

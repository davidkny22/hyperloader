"""Canonical JSON model for pinned Torch compatibility streams."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

FORMAT = "hyperloader.torch-golden"


def canonical_system(value: str) -> str:
    """Return the release-platform name for an operating-system label."""
    normalized = value.strip().lower()
    return "macos" if normalized == "darwin" else normalized


def encode_value(value: Any) -> dict[str, Any]:
    """Encode one result with structural identity and exact numeric bits."""
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        contiguous = tensor.contiguous()
        bits = contiguous.view(torch.uint8).numpy().tobytes().hex()
        return {
            "kind": "tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "layout": str(tensor.layout),
            "bits": bits,
        }
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "type": _type_name(value),
            "items": [
                [encode_value(key), encode_value(item)] for key, item in value.items()
            ],
        }
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return {
            "kind": "namedtuple",
            "type": _type_name(value),
            "items": [encode_value(item) for item in value],
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": "sequence",
            "type": _type_name(value),
            "items": [encode_value(item) for item in value],
        }
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": str(value)}
    if isinstance(value, float):
        return {"kind": "float", "bits": struct.pack(">d", value).hex()}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, bytes):
        return {"kind": "bytes", "value": value.hex()}
    raise TypeError(f"unsupported golden value type: {_type_name(value)}")


def validate_document(document: object) -> dict[str, Any]:
    """Validate the stable envelope needed by the parity verifier."""
    if not isinstance(document, dict):
        raise TypeError("torch golden document must be a dictionary")
    required = {"format", "environment", "cases"}
    if set(document) != required or document.get("format") != FORMAT:
        raise ValueError("torch golden document has an invalid envelope")
    environment = document["environment"]
    if not isinstance(environment, dict):
        raise TypeError("torch golden environment must be a dictionary")
    environment_fields = {
        "torch",
        "torch_minor",
        "python",
        "implementation",
        "system",
        "release",
        "machine",
        "multiprocessing_start_method",
        "in_order",
    }
    if set(environment) != environment_fields:
        raise ValueError("torch golden environment fields do not match the schema")
    cases = document["cases"]
    if not isinstance(cases, dict) or not cases:
        raise ValueError("torch golden document must contain named cases")
    for name, streams in cases.items():
        if not isinstance(name, str) or not isinstance(streams, list) or not streams:
            raise ValueError("torch golden cases must contain nonempty stream lists")
        if not all(isinstance(stream, list) for stream in streams):
            raise TypeError("each torch golden epoch must be a list")
    return document


def write_document(path: Path, document: dict[str, Any]) -> str:
    """Write canonical JSON and return its SHA-256 digest."""
    validate_document(document)
    encoded = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def read_document(path: Path) -> dict[str, Any]:
    """Read and validate one canonical golden artifact."""
    return validate_document(json.loads(path.read_text(encoding="utf-8")))


def _type_name(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"

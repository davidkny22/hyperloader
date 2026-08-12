"""Canonical machine-readable training-evaluation output."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_VISUAL_SUFFIXES = {".gif", ".html", ".jpeg", ".jpg", ".pdf", ".png", ".svg", ".webp"}


def write_result(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write canonical JSON and reject visualization targets."""
    if path.suffix.lower() in _VISUAL_SUFFIXES:
        raise ValueError("training evaluation emits machine-readable records only")
    if path.suffix.lower() != ".json":
        raise ValueError("training evaluation output must use a .json path")
    document = {"schema_version": SCHEMA_VERSION, **payload}
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise

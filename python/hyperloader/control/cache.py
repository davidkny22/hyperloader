"""Calibration cache paths, persistence, and machine invalidation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .machine import MachineIdentity
from .record import CalibrationRecord


def calibration_cache_path(root: Path, machine: MachineIdentity) -> Path:
    """Return the opaque machine-keyed calibration path."""
    return root / "calibration" / f"{machine.cache_key}.json"


def save_calibration(record: CalibrationRecord, path: Path) -> None:
    """Atomically persist one validated calibration record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_calibration(path: Path, machine: MachineIdentity) -> CalibrationRecord | None:
    """Load a matching record or invalidate it when machine identity changed."""
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("calibration record must be a JSON object")
    record = CalibrationRecord.from_dict(raw)
    return record if record.machine.cache_key == machine.cache_key else None

"""Dataset-root-relative file inventory and strict digest elements."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from .model import FingerprintElement


def file_elements(
    raw_paths: list[Any], mode: str, prefix: str, explicit_root: Any = None
) -> list[FingerprintElement]:
    """Describe ordered file names and sizes without observing mtimes."""
    paths = [_coerce_path(raw) for raw in raw_paths]
    root = _dataset_root(paths, explicit_root)
    elements = [FingerprintElement(f"{prefix}.root_kind", "dataset-relative")]
    elements.append(FingerprintElement(f"{prefix}.count", len(paths)))
    for index, path in enumerate(paths):
        item_prefix = f"{prefix}[{index}]"
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = os.path.relpath(path, root).replace(os.sep, "/")
        try:
            size: int | None = path.stat().st_size
        except OSError:
            size = None
        elements.append(FingerprintElement(f"{item_prefix}.path", relative))
        elements.append(FingerprintElement(f"{item_prefix}.size", size))
        if mode == "strict":
            digest = _file_digest(path) if size is not None else "missing"
            elements.append(FingerprintElement(f"{item_prefix}.sha256", digest))
    return elements


def _dataset_root(paths: list[Path], explicit_root: Any) -> Path:
    if explicit_root is not None:
        return _coerce_path(explicit_root)
    if not paths:
        return Path.cwd().resolve()
    parents = [str(path.parent) for path in paths]
    try:
        return Path(os.path.commonpath(parents)).resolve()
    except ValueError:
        return paths[0].parent


def _coerce_path(value: Any) -> Path:
    try:
        return Path(os.fsdecode(os.fspath(value))).resolve(strict=False)
    except TypeError:
        return Path(str(value)).resolve(strict=False)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

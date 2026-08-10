"""Release-authored per-platform decoder pin table."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


def platform_pin(platform: str, codec: str) -> tuple[str, str]:
    """Return the static backend and version for one supported codec."""
    document = _load_table()
    platforms = document["platforms"]
    if platform not in platforms:
        raise ValueError(
            f"decoder substitution has no static table for platform {platform}"
        )
    table = platforms[platform]
    if codec not in table:
        raise ValueError(f"decoder substitution does not support codec {codec}")
    row = table[codec]
    return row["backend"], row["version"]


def _load_table() -> dict[str, Any]:
    payload = files(__package__).joinpath("pins.json").read_text(encoding="utf-8")
    document = json.loads(payload)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise RuntimeError("decoder pins must use schema version 1")
    platforms = document.get("platforms")
    if not isinstance(platforms, dict):
        raise RuntimeError("decoder pins must contain a platform table")
    return document

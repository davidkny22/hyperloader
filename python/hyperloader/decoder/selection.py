"""Resolve pipeline decoder declarations to exact selected pins."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

from hyperloader.config import AUTO
from hyperloader.stages import Decode, Pipeline

from ..fingerprint.callable import callable_identity
from .model import DecoderSelection
from .pins import platform_pin


def select_decoder_pins(
    dataset: Any, configured: object, *, platform: str | None = None
) -> tuple[DecoderSelection, ...]:
    """Select every pipeline decoder without probing runtime performance."""
    platform = sys.platform if platform is None else platform
    overrides = _overrides(configured)
    selections = []
    consumed = set()
    if isinstance(dataset, Pipeline):
        for index, stage in enumerate(dataset.sample_stages):
            if not isinstance(stage, Decode):
                continue
            stage_name = f"pipeline-decode-{index}"
            if stage.substitute:
                key = stage_name if stage_name in overrides else stage.codec
                if key in overrides:
                    backend, version = overrides[key]
                    consumed.add(key)
                    source = "configured"
                else:
                    backend, version = platform_pin(platform, stage.codec)
                    source = "platform-table"
                selections.append(
                    DecoderSelection(
                        stage_name,
                        stage.codec,
                        backend,
                        version,
                        platform,
                        source,
                        True,
                    )
                )
            else:
                identity = callable_identity(stage.fn)
                version = str(
                    identity.get("code_sha256") or identity.get("call_sha256") or "none"
                )
                selections.append(
                    DecoderSelection(
                        stage_name,
                        stage.codec or "custom",
                        str(identity.get("qualname", "user-callable")),
                        version,
                        platform,
                        "user-callable",
                        False,
                    )
                )
    unused = set(overrides) - consumed
    if unused:
        names = ", ".join(sorted(unused))
        raise ValueError(
            f"decoder pin overrides did not match substituted stages: {names}"
        )
    return tuple(selections)


def _overrides(configured: object) -> dict[str, tuple[str, str]]:
    if configured is AUTO:
        return {}
    if not isinstance(configured, Mapping):
        raise TypeError("determinism.decoder_pins must be a mapping or auto")
    return {str(key): _pin_value(value) for key, value in configured.items()}


def _pin_value(value: object) -> tuple[str, str]:
    if isinstance(value, str):
        backend, separator, version = value.rpartition("@")
        if separator and backend and version:
            return backend, version
    if isinstance(value, Mapping):
        backend = value.get("backend")
        version = value.get("version")
        if (
            isinstance(backend, str)
            and backend
            and isinstance(version, str)
            and version
        ):
            return backend, version
    raise ValueError("decoder pin values must name a nonempty backend and version")

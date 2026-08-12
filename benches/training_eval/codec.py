"""JSON decoding for live-training protocol records."""

from __future__ import annotations

from typing import Any

from .models import (
    DecisionRule,
    TrainingCellConfig,
    TrainingEnvironment,
    TrainingHalf,
    TrainingObservation,
)


def decode_config(document: dict[str, Any]) -> TrainingCellConfig:
    """Decode one point configuration from JSON-compatible values."""
    values = dict(document)
    values["tuning_knobs"] = tuple(values["tuning_knobs"])
    values["decision"] = DecisionRule(**values["decision"])
    return TrainingCellConfig(**values)


def decode_half(document: dict[str, Any]) -> TrainingHalf:
    """Decode one timed half and its environment."""
    values = dict(document)
    values["environment"] = TrainingEnvironment(**values["environment"])
    return TrainingHalf(**values)


def decode_observation(document: dict[str, Any]) -> TrainingObservation:
    """Decode one complete feeder-swap observation."""
    return TrainingObservation(
        ordinal=int(document["ordinal"]),
        config=decode_config(document["config"]),
        first=decode_half(document["first"]),
        second=decode_half(document["second"]),
        uninterrupted_model_process=bool(document["uninterrupted_model_process"]),
    )

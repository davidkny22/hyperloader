"""Data-driven dataset-type registration and plan selection."""

from __future__ import annotations

import json
from importlib import import_module
from importlib.resources import files
from typing import Any

from .black_box import BlackBoxPlan, build_black_box_plan
from .stages import StagePlan, build_stage_plan
from .structured import StructurePlan
from .tensor import TensorPlan
from ..stages import Pipeline

Plan = BlackBoxPlan | StagePlan | StructurePlan | TensorPlan


def _load_mappings() -> tuple[dict[str, str], ...]:
    payload = files(__package__).joinpath("mappings.json").read_text(encoding="utf-8")
    document = json.loads(payload)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise RuntimeError("planner mappings must use schema version 1")
    rows = document.get("mappings")
    if not isinstance(rows, list):
        raise RuntimeError("planner mappings must contain a list")
    return tuple(rows)


def _registered_builder(dataset: Any) -> str | None:
    dataset_type = type(dataset)
    for row in _load_mappings():
        candidates = (
            (dataset_type,)
            if row.get("match", "exact") == "exact"
            else dataset_type.__mro__
        )
        if any(
            row["dataset_module"] == candidate.__module__
            and row["dataset_type"] == candidate.__name__
            for candidate in candidates
        ):
            return row["builder"]
    return None


def _build_registered(dataset: Any, shuffle: bool | None, path: str) -> Plan | None:
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        return None
    builder = getattr(import_module(module_name), attribute)
    return builder(dataset, shuffle)


def build_plan(dataset: Any, shuffle: bool | None) -> Plan | None:
    """Select a registered plan or the non-erroring black-box refuge."""
    if isinstance(dataset, Pipeline):
        return build_stage_plan(dataset, shuffle)
    try:
        builder = _registered_builder(dataset)
        if builder is not None:
            plan = _build_registered(dataset, shuffle, builder)
            if plan is not None:
                return plan
    except Exception:
        pass
    return build_black_box_plan(dataset, shuffle)

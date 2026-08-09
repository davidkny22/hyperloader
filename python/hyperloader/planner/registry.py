"""Data-driven dataset-type registration and plan selection."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from .black_box import BlackBoxPlan, build_black_box_plan
from .tensor import TensorPlan, build_tensor_plan

Plan = BlackBoxPlan | TensorPlan


def _load_mappings() -> tuple[dict[str, str], ...]:
    payload = files(__package__).joinpath("mappings.json").read_text(encoding="utf-8")
    rows = json.loads(payload)
    if not isinstance(rows, list):
        raise RuntimeError("planner mappings must be a list")
    return tuple(rows)


def _registered_plan(dataset: Any) -> str | None:
    dataset_type = type(dataset)
    for row in _load_mappings():
        if (
            row["dataset_module"] == dataset_type.__module__
            and row["dataset_type"] == dataset_type.__name__
        ):
            return row["plan"]
    return None


def build_plan(dataset: Any, shuffle: bool | None) -> Plan | None:
    """Select a registered plan or the non-erroring black-box refuge."""
    registration = _registered_plan(dataset)
    if registration == "contiguous_tensor":
        return build_tensor_plan(dataset, shuffle)
    if registration is not None:
        raise RuntimeError(f"unknown planner registration {registration!r}")
    return build_black_box_plan(dataset, shuffle)

"""Public loader diagnosis entry point."""

from __future__ import annotations

from typing import Any

from .attribution import attribute_cause
from .model import DiagnosisReport
from .native import observe_native
from .probe import run_probe
from .promotion import promotion_record
from .render import render_diagnosis
from .stock import observe_stock


def diagnose(
    loader: Any, *, probe: bool = False, probe_batches: int = 4
) -> DiagnosisReport:
    """Explain one stock or hyperloader data path from observable evidence."""
    loader_kind = _loader_kind(loader)
    active_probe = run_probe(loader, probe_batches) if probe else None
    record = (
        observe_native(loader) if loader_kind == "hyperloader" else observe_stock(loader)
    )
    record.update(
        {
            "schema": "hyperloader.diagnosis/1",
            "observation_mode": "active-probe" if probe else "passive",
            "probe": active_probe,
        }
    )
    if active_probe is not None:
        record["gil_release"] = {
            "basis": (
                "probe-side Python thread progress divided by its equal-duration "
                "sleep calibration"
            ),
            "fraction": active_probe["gil_release_fraction"],
        }
    record["attribution"] = attribute_cause(record)
    record["promotion"] = promotion_record(
        loader,
        loader_kind=loader_kind,
        saturation=record["saturation"],
        gil_release=record["gil_release"],
    )
    return DiagnosisReport(render_diagnosis(record), record)


def _loader_kind(loader: Any) -> str:
    if callable(getattr(loader, "stats", None)) and hasattr(loader, "config"):
        return "hyperloader"
    torch_loader = any(
        base.__module__.startswith("torch.utils.data")
        for base in type(loader).__mro__
    )
    if torch_loader and hasattr(loader, "dataset"):
        return "torch"
    raise TypeError("diagnose expects a hyperloader or torch DataLoader")

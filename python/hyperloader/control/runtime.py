"""Loader-specific controller construction and report serialization."""

from __future__ import annotations

from typing import Any

from hyperloader.config import AUTO

from .cache import calibration_cache_path, load_calibration, user_cache_root
from .controller import AdaptiveController, ControllerDecision
from .machine import detect_machine_identity
from .objective import ControllerObjective
from .priors import spark_prior


def resolve_calibration() -> Any:
    """Load the current machine's record or its narrowly matched measured prior."""
    machine = detect_machine_identity()
    path = calibration_cache_path(user_cache_root(), machine)
    try:
        cached = load_calibration(path, machine)
    except (OSError, ValueError):
        cached = None
    return cached if cached is not None else spark_prior(machine)


def build_controller(loader: Any) -> AdaptiveController:
    """Construct a plan-local controller at the spawned worker ceiling."""
    calibration = resolve_calibration()
    loader._calibration = calibration
    configured = loader.config.control.cadence
    cadence_seconds = (
        loader.config.factors.f_cad_s if configured is AUTO else float(configured)
    )
    cadence_batches = loader.config.factors.f_cad_b if configured is AUTO else 1
    return AdaptiveController(
        width_ceiling=loader._process_pool.worker_count,
        cadence_seconds=cadence_seconds,
        cadence_batches=cadence_batches,
        step_clip=loader.config.factors.step_clip,
        shrink_hysteresis=loader.config.factors.hysteresis,
        objective=ControllerObjective(calibration),
    )


def decision_report(decision: ControllerDecision) -> dict[str, int | float | str | bool]:
    """Return a stable report payload for telemetry and gate assurance."""
    return {
        "previous_width": decision.previous_width,
        "reason": decision.reason,
        "resource_loss": decision.score[1],
        "starvation": decision.starvation,
        "width": decision.width,
    }

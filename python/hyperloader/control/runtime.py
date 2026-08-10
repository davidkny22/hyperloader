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
    spawned_ceiling = loader._process_pool.worker_count
    configured_cpu = loader.config.control.ceilings.cpu_cores
    if configured_cpu is AUTO:
        width_ceiling = spawned_ceiling
        cpu_ceiling_binding = False
    else:
        if configured_cpu == 0:
            raise ValueError(
                "control.ceilings.cpu_cores must be positive for process execution"
            )
        width_ceiling = min(spawned_ceiling, int(configured_cpu))
        cpu_ceiling_binding = width_ceiling < spawned_ceiling
    configured_bandwidth = loader.config.control.ceilings.bandwidth
    bandwidth_ceiling = (
        None
        if configured_bandwidth is AUTO
        else float(configured_bandwidth) * 1_000_000_000.0
    )
    return AdaptiveController(
        width_ceiling=width_ceiling,
        cadence_seconds=cadence_seconds,
        cadence_batches=cadence_batches,
        step_clip=loader.config.factors.step_clip,
        shrink_hysteresis=loader.config.factors.hysteresis,
        objective=ControllerObjective(calibration),
        cpu_ceiling_binding=cpu_ceiling_binding,
        bandwidth_ceiling=bandwidth_ceiling,
    )


def decision_report(
    decision: ControllerDecision,
) -> dict[str, int | float | str | bool | None]:
    """Return a stable report payload for telemetry and gate assurance."""
    return {
        "binding": decision.binding,
        "previous_width": decision.previous_width,
        "reason": decision.reason,
        "resource_loss": decision.score[1],
        "starvation": decision.starvation,
        "width": decision.width,
    }

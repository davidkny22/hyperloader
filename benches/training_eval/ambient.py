"""Steady-machine acceptance from pre-session ambient probes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AmbientProbe:
    """One fixed-work probe with raw activity components retained."""

    probe_id: str
    rate: float
    duration_seconds: float
    cpu_percent: float
    accelerator_percent: float
    memory_used_bytes: int


@dataclass(frozen=True)
class AmbientDecision:
    """Current-to-prior ambient rate difference against the null band."""

    status: str
    delta_percent: float
    null_band_percent: float
    prior_probe_id: str
    current_probe_id: str


def compare_ambient(
    prior: AmbientProbe,
    current: AmbientProbe,
    *,
    null_band_percent: float,
) -> AmbientDecision:
    """Accept a session only when its fixed probe remains inside the null band."""
    if prior.rate <= 0 or current.rate <= 0 or prior.duration_seconds <= 0:
        raise ValueError("ambient probes require positive rates and durations")
    if current.duration_seconds != prior.duration_seconds:
        raise ValueError("ambient probe durations must match")
    if null_band_percent < 0:
        raise ValueError("null band must be nonnegative")
    delta = 100.0 * (current.rate - prior.rate) / prior.rate
    return AmbientDecision(
        status="pass" if abs(delta) <= null_band_percent else "fail",
        delta_percent=delta,
        null_band_percent=null_band_percent,
        prior_probe_id=prior.probe_id,
        current_probe_id=current.probe_id,
    )

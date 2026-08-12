"""Causal diagnosis from observable loader signals."""

from __future__ import annotations


def attribute_cause(observation: dict[str, object]) -> dict[str, object]:
    """Select the strongest supported cause without filling evidence gaps."""
    bindings = observation["ceiling_binds"]
    if isinstance(bindings, list) and bindings:
        return {
            "cause": "user_ceiling",
            "basis": f"The active controller named {bindings[0]} as binding.",
        }
    blocking = observation["blocking"]
    if isinstance(blocking, dict):
        fraction = blocking.get("fraction")
        if isinstance(fraction, (float, int)) and fraction > 0:
            return {
                "cause": "delivery_wait",
                "basis": "Measured loader wait time is nonzero.",
            }
        if blocking.get("currently_blocked") is True:
            return {
                "cause": "delivery_wait",
                "basis": "Work is outstanding while the ready-result queue is empty.",
            }
    saturation = observation["saturation"]
    if isinstance(saturation, dict):
        occupancy = saturation.get("occupancy_fraction")
        if isinstance(occupancy, (float, int)) and occupancy < 0.5:
            return {
                "cause": "low_ready_saturation",
                "basis": "Ready occupancy is below half of the observed capacity.",
            }
    return {
        "cause": "not_identified",
        "basis": "The passive signals do not support a unique bottleneck cause.",
    }

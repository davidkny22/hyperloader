"""Measured machine-class priors used before local calibration exists."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any

from .machine import MachineIdentity
from .record import CalibrationRecord


@lru_cache(maxsize=1)
def _spark_profile() -> dict[str, Any]:
    """Load the shipped Spark calibration profile."""
    resource = files(__package__).joinpath("spark_prior.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Spark calibration profile must be an object")
    return payload


def spark_prior(machine: MachineIdentity) -> CalibrationRecord | None:
    """Return the Spark campaign prior only for the measured hardware class."""
    profile = _spark_profile()
    match = profile["match"]
    calibration = profile["calibration"]
    if not isinstance(match, dict) or not isinstance(calibration, dict):
        raise ValueError("Spark calibration profile sections must be objects")

    model = machine.cpu_model.casefold()
    model_markers = tuple(str(value).casefold() for value in match["model_markers"])
    logical_cpus = {cpu for cluster in machine.clusters for cpu in cluster.logical_cpus}
    frequencies = {
        cluster.max_frequency_hz
        for cluster in machine.clusters
        if cluster.max_frequency_hz is not None
    }
    memory_range = tuple(int(value) for value in match["memory_range_bytes"])
    detected_spark_topology = (
        model == str(match["kernel_architecture"]).casefold()
        and len(logical_cpus) == int(match["logical_cpus"])
        and int(match["efficiency_frequency_hz"]) in frequencies
        and int(match["performance_frequency_hz"]) in frequencies
        and memory_range[0] <= machine.memory_bytes <= memory_range[1]
    )
    if not (
        model_markers[0] in model
        or all(marker in model for marker in model_markers[1:])
        or detected_spark_topology
    ):
        return None

    record = dict(calibration)
    record["machine"] = machine.to_dict()
    record["machine_key"] = machine.cache_key
    return CalibrationRecord.from_dict(record)

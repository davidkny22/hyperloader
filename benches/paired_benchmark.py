"""Validate paired observation JSON and emit the protocol decision."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmark_protocol import (
    CommonConfig,
    EnvironmentMetadata,
    PairedObservation,
    SystemRun,
    TuningBudget,
    evaluate,
)


def decode_run(document: dict[str, Any]) -> SystemRun:
    """Decode one immutable system run from a JSON object."""
    return SystemRun(
        system=document["system"],
        throughput=float(document["throughput"]),
        duration_seconds=float(document["duration_seconds"]),
        warmed=bool(document["warmed"]),
        config=CommonConfig(**document["config"]),
        tuning=TuningBudget(
            trials=int(document["tuning"]["trials"]),
            wall_seconds=float(document["tuning"]["wall_seconds"]),
            knobs=tuple(document["tuning"]["knobs"]),
        ),
        environment=EnvironmentMetadata(**document["environment"]),
    )


def decode_observation(document: dict[str, Any]) -> PairedObservation:
    """Decode one feeder-swap observation from a JSON object."""
    return PairedObservation(
        ordinal=int(document["ordinal"]),
        first=decode_run(document["first"]),
        second=decode_run(document["second"]),
        uninterrupted=bool(document["uninterrupted"]),
    )


def main() -> None:
    """Read JSONL records and write one deterministic decision document."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--threshold-percent", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    observations = [
        decode_observation(json.loads(line))
        for line in arguments.observations.read_text(encoding="utf-8").splitlines()
        if line
    ]
    result = evaluate(observations, threshold_percent=arguments.threshold_percent)
    arguments.output.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

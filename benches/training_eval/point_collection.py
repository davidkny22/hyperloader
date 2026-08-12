"""Durable collection loop shared by live training point types."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from .decision import TrainingDecision, decide
from .live_cell import (
    BatchFeeder,
    StepRunner,
    run_training_observation,
    warm_training_process,
)
from .models import TrainingCellConfig, TrainingEnvironment, TrainingObservation
from .output import write_result


def collect_point(
    config: TrainingCellConfig,
    environment: TrainingEnvironment,
    *,
    feeders: Mapping[str, BatchFeeder],
    runner: StepRunner,
    warmup_steps: int,
    observations_path: Path,
    decision_path: Path,
) -> TrainingDecision:
    """Collect alternating pairs until the preregistered decision is terminal."""
    if warmup_steps <= 0:
        raise ValueError("warmup size must be positive")
    if observations_path.exists() or decision_path.exists():
        raise FileExistsError("training point output already exists")
    observations_path.parent.mkdir(parents=True, exist_ok=True)
    process_token = uuid.uuid4().hex
    optimizer_step = warm_training_process(
        feeders,
        runner,
        feeder_order=(config.reference, config.subject),
        steps_per_feeder=warmup_steps,
    )
    observations: list[TrainingObservation] = []
    hash_chain = "0" * 64
    try:
        while True:
            observation = run_training_observation(
                config,
                environment,
                ordinal=len(observations),
                process_token=process_token,
                optimizer_step_start=optimizer_step,
                feeders=feeders,
                runner=runner,
                warmup_complete=True,
                initial_hash_chain=hash_chain,
            )
            observations.append(observation)
            optimizer_step = observation.second.optimizer_step_stop
            hash_chain = observation.second.batch_hash_chain
            _append_observation(observations_path, observation)
            result = decide(observations)
            if result.status != "collect":
                write_result(
                    decision_path,
                    {
                        "kind": "training-throughput-decision",
                        "evaluation_id": config.evaluation_id,
                        "point_id": config.point_id,
                        "config": asdict(config),
                        "decision": asdict(result),
                        "observations": str(observations_path.name),
                    },
                )
                return result
    finally:
        for feeder in feeders.values():
            close = getattr(feeder, "close", None)
            if close is not None:
                close()


def _append_observation(path: Path, observation: TrainingObservation) -> None:
    encoded = json.dumps(asdict(observation), sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

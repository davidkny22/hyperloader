"""Collection of one live-training token comparison point."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from .decision import TrainingDecision, decide
from .feeders import ResidentTokenFeeder, TokenBatch, collate_token_batch
from .live_cell import BatchFeeder, run_training_observation, warm_training_process
from .models import TrainingCellConfig, TrainingEnvironment, TrainingObservation
from .output import write_result
from .public_feeders import PublicLoaderFeeder, build_public_feeder
from .token_source import PretokenizedRows
from .training_step import TransformerStepRunner


def collect_token_point(
    config: TrainingCellConfig,
    environment: TrainingEnvironment,
    *,
    model_factory: Callable[[], nn.Module],
    device: torch.device,
    seed: int,
    bank_batches: int,
    warmup_steps: int,
    pin_memory: bool,
    observations_path: Path,
    decision_path: Path,
) -> TrainingDecision:
    """Collect alternating pairs until the preregistered decision is terminal."""
    if bank_batches <= 0 or warmup_steps <= 0:
        raise ValueError("resident bank and warmup sizes must be positive")
    torch.manual_seed(seed)
    model = model_factory()
    vocabulary_size = int(getattr(model, "vocabulary_size", 0))
    if vocabulary_size <= 1:
        raise ValueError("token models must expose a vocabulary_size greater than one")
    dataset = PretokenizedRows(
        rows=bank_batches * config.batch_size,
        sequence_length=config.sequence_length,
        vocabulary_size=vocabulary_size,
        seed=seed,
    )
    resident = _resident_batches(
        dataset,
        batch_size=config.batch_size,
        pin_memory=pin_memory,
    )
    feeders = _build_feeders(
        config,
        dataset,
        resident,
        pin_memory=pin_memory,
    )
    runner = TransformerStepRunner(
        model,
        device=device,
        precision=config.precision,
        learning_rate=3e-4,
        non_blocking=pin_memory,
    )
    process_token = uuid.uuid4().hex
    optimizer_step = warm_training_process(
        feeders,
        runner,
        feeder_order=(config.reference, config.subject),
        steps_per_feeder=warmup_steps,
    )
    observations: list[TrainingObservation] = []
    hash_chain = "0" * 64
    if observations_path.exists() or decision_path.exists():
        raise FileExistsError("training point output already exists")
    observations_path.parent.mkdir(parents=True, exist_ok=True)
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


def _resident_batches(
    dataset: PretokenizedRows,
    *,
    batch_size: int,
    pin_memory: bool,
) -> tuple[TokenBatch, ...]:
    batches: list[TokenBatch] = []
    for start in range(0, len(dataset), batch_size):
        batch = collate_token_batch(
            [dataset[index] for index in range(start, start + batch_size)]
        )
        if pin_memory:
            batch = TokenBatch(batch.tokens.pin_memory(), batch.digest)
        batches.append(batch)
    return tuple(batches)


def _build_feeders(
    config: TrainingCellConfig,
    dataset: PretokenizedRows,
    resident: tuple[TokenBatch, ...],
    *,
    pin_memory: bool,
) -> dict[str, BatchFeeder]:
    return {
        system: _build_feeder(
            system,
            config,
            dataset,
            resident,
            pin_memory=pin_memory,
        )
        for system in (config.reference, config.subject)
    }


def _build_feeder(
    system: str,
    config: TrainingCellConfig,
    dataset: PretokenizedRows,
    resident: tuple[TokenBatch, ...],
    *,
    pin_memory: bool,
) -> ResidentTokenFeeder | PublicLoaderFeeder:
    if system == "counterfactual" or system.startswith("null-"):
        return ResidentTokenFeeder(system, resident)
    workers = (
        config.subject_workers if system == config.subject else config.reference_workers
    )
    prefetch = (
        config.subject_prefetch
        if system == config.subject
        else config.reference_prefetch
    )
    return build_public_feeder(
        system,
        dataset,
        batch_size=config.batch_size,
        workers=workers,
        prefetch=prefetch,
        collate=collate_token_batch,
        pin_memory=pin_memory,
    )


def _append_observation(path: Path, observation: TrainingObservation) -> None:
    encoded = json.dumps(asdict(observation), sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

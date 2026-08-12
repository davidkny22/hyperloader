"""Collection of one live-training token comparison point."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from .decision import TrainingDecision
from .feeders import ResidentTokenFeeder, TokenBatch, collate_token_batch
from .live_cell import BatchFeeder
from .models import TrainingCellConfig, TrainingEnvironment
from .point_collection import collect_point
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
    if config.sequence_length is None:
        raise ValueError("token points require a sequence length")
    dataset = PretokenizedRows(
        rows=bank_batches * config.batch_size,
        sequence_length=config.sequence_length,
        vocabulary_size=vocabulary_size,
        seed=seed,
    )
    if dataset.identity != config.dataset_identity:
        raise ValueError("token source does not match the recorded dataset identity")
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
        learning_rate=config.learning_rate,
        non_blocking=pin_memory,
    )
    return collect_point(
        config,
        environment,
        feeders=feeders,
        runner=runner,
        warmup_steps=warmup_steps,
        observations_path=observations_path,
        decision_path=decision_path,
    )


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

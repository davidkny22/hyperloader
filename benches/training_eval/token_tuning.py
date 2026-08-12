"""Live-step tuning composition for token training feeders."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch import nn

from .feeders import collate_token_batch
from .public_feeders import build_public_feeder
from .token_source import PretokenizedRows
from .training_step import TransformerStepRunner
from .tuning import TuningCandidate, tune_live_feeder


def tune_token_system(
    system: str,
    *,
    dataset: PretokenizedRows,
    batch_size: int,
    model_factory: Callable[[], nn.Module],
    candidates: Sequence[TuningCandidate],
    device: torch.device,
    precision: str,
    learning_rate: float,
    pin_memory: bool,
    seed: int,
    seconds_per_trial: float,
    warmup_steps: int,
    output: Path,
) -> TuningCandidate:
    """Tune one installed public token feeder against real optimizer steps."""

    def build(candidate: TuningCandidate):
        torch.manual_seed(seed)
        feeder = build_public_feeder(
            system,
            dataset,
            batch_size=batch_size,
            workers=candidate.workers,
            prefetch=candidate.prefetch,
            collate=collate_token_batch,
            pin_memory=pin_memory,
        )
        runner = TransformerStepRunner(
            model_factory(),
            device=device,
            precision=precision,
            learning_rate=learning_rate,
            non_blocking=pin_memory,
        )
        return feeder, runner

    return tune_live_feeder(
        system,
        candidates,
        build_trial=build,
        seconds_per_trial=seconds_per_trial,
        warmup_steps=warmup_steps,
        output=output,
    )

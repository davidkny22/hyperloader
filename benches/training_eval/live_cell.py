"""Uninterrupted paired feeder swaps for live optimizer execution."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from .feeders import TokenBatch
from .hash_chain import EMPTY_HASH_CHAIN, advance_hash_chain
from .models import (
    TrainingCellConfig,
    TrainingEnvironment,
    TrainingHalf,
    TrainingObservation,
)


class BatchFeeder(Protocol):
    """Supply pre-tokenized batches under one named feeder identity."""

    system: str

    def next_batch(self) -> TokenBatch:
        """Return one batch without restarting the model process."""


class StepRunner(Protocol):
    """Execute and settle one real training step."""

    def step(self, batch: TokenBatch) -> Any:
        """Launch forward, backward, and optimizer work."""

    def finish(self, loss: Any) -> float:
        """Settle outstanding work and return the terminal loss."""


def warm_training_process(
    feeders: Mapping[str, BatchFeeder],
    runner: StepRunner,
    *,
    feeder_order: tuple[str, ...],
    steps_per_feeder: int,
) -> int:
    """Warm every feeder outside timing while preserving one model and optimizer."""
    if steps_per_feeder <= 0 or set(feeder_order) != set(feeders):
        raise ValueError("warmup requires every feeder and a positive step count")
    terminal_loss: Any | None = None
    steps = 0
    for system in feeder_order:
        for _ in range(steps_per_feeder):
            terminal_loss = runner.step(feeders[system].next_batch())
            steps += 1
    assert terminal_loss is not None
    runner.finish(terminal_loss)
    return steps


def run_training_observation(
    config: TrainingCellConfig,
    environment: TrainingEnvironment,
    *,
    ordinal: int,
    process_token: str,
    optimizer_step_start: int,
    feeders: Mapping[str, BatchFeeder],
    runner: StepRunner,
    warmup_complete: bool,
    initial_hash_chain: str = EMPTY_HASH_CHAIN,
    clock: Callable[[], float] = time.perf_counter,
) -> TrainingObservation:
    """Run two contiguous timed halves through one model and optimizer instance."""
    if not warmup_complete:
        raise ValueError("live training observations require completed untimed warmup")
    if ordinal < 0 or optimizer_step_start < 0 or not process_token:
        raise ValueError("ordinal, optimizer step, and process token are invalid")
    required = {config.subject, config.reference}
    if set(feeders) != required:
        raise ValueError("feeders must match the declared subject and reference")
    first_system = config.reference if ordinal % 2 == 0 else config.subject
    second_system = config.subject if ordinal % 2 == 0 else config.reference
    first = _run_half(
        feeders[first_system],
        config,
        environment,
        process_token=process_token,
        optimizer_step_start=optimizer_step_start,
        runner=runner,
        initial_hash_chain=initial_hash_chain,
        clock=clock,
    )
    second = _run_half(
        feeders[second_system],
        config,
        environment,
        process_token=process_token,
        optimizer_step_start=first.optimizer_step_stop,
        runner=runner,
        initial_hash_chain=first.batch_hash_chain,
        clock=clock,
    )
    return TrainingObservation(
        ordinal=ordinal,
        config=config,
        first=first,
        second=second,
        uninterrupted_model_process=True,
    )


def _run_half(
    feeder: BatchFeeder,
    config: TrainingCellConfig,
    environment: TrainingEnvironment,
    *,
    process_token: str,
    optimizer_step_start: int,
    runner: StepRunner,
    initial_hash_chain: str,
    clock: Callable[[], float],
) -> TrainingHalf:
    started = clock()
    deadline = started + config.half_seconds
    steps = 0
    samples = 0
    chain = initial_hash_chain
    terminal_loss: Any | None = None
    while steps == 0 or clock() < deadline:
        batch = feeder.next_batch()
        terminal_loss = runner.step(batch)
        chain = advance_hash_chain(chain, batch.digest)
        samples += batch.samples
        steps += 1
    assert terminal_loss is not None
    loss = runner.finish(terminal_loss)
    duration = clock() - started
    if duration <= 0:
        raise RuntimeError("training half elapsed time must be positive")
    return TrainingHalf(
        system=feeder.system,
        process_token=process_token,
        duration_seconds=duration,
        optimizer_step_start=optimizer_step_start,
        optimizer_step_stop=optimizer_step_start + steps,
        samples=samples,
        rate_steps_per_second=steps / duration,
        rate_samples_per_second=samples / duration,
        warmed=True,
        batch_hash_chain=chain,
        terminal_loss=loss,
        environment=environment,
    )

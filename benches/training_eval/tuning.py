"""Equal-budget live-step tuning for public training feeders."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .live_cell import BatchFeeder, StepRunner
from .output import write_result


@dataclass(frozen=True)
class TuningCandidate:
    """One worker and prefetch setting in a caller-supplied search set."""

    workers: int
    prefetch: int

    def validate(self) -> None:
        """Reject controls that cannot construct a positive-worker feeder."""
        if self.workers <= 0 or self.prefetch <= 0:
            raise ValueError("tuning workers and prefetch must be positive")


@dataclass(frozen=True)
class TuningTrial:
    """Measured live-step rate for one candidate."""

    candidate: TuningCandidate
    steps: int
    samples: int
    elapsed_seconds: float
    steps_per_second: float
    samples_per_second: float


def tune_live_feeder(
    system: str,
    candidates: Sequence[TuningCandidate],
    *,
    build_trial: Callable[[TuningCandidate], tuple[BatchFeeder, StepRunner]],
    seconds_per_trial: float,
    warmup_steps: int,
    output: Path,
    clock: Callable[[], float] = time.perf_counter,
) -> TuningCandidate:
    """Select the highest live sample rate under one equal candidate budget."""
    if not system or not candidates or seconds_per_trial <= 0 or warmup_steps <= 0:
        raise ValueError("tuning requires a system, candidates, time, and warmup")
    if len(set(candidates)) != len(candidates):
        raise ValueError("tuning candidates must be unique")
    trials = []
    for candidate in candidates:
        candidate.validate()
        feeder, runner = build_trial(candidate)
        try:
            terminal_loss: Any | None = None
            for _ in range(warmup_steps):
                terminal_loss = runner.step(feeder.next_batch())
            assert terminal_loss is not None
            runner.finish(terminal_loss)
            started = clock()
            deadline = started + seconds_per_trial
            steps = 0
            samples = 0
            while steps == 0 or clock() < deadline:
                batch = feeder.next_batch()
                terminal_loss = runner.step(batch)
                steps += 1
                samples += batch.samples
            runner.finish(terminal_loss)
            elapsed = clock() - started
        finally:
            close = getattr(feeder, "close", None)
            if close is not None:
                close()
        trials.append(
            TuningTrial(
                candidate,
                steps,
                samples,
                elapsed,
                steps / elapsed,
                samples / elapsed,
            )
        )
    winner = max(trials, key=lambda trial: trial.samples_per_second)
    write_result(
        output,
        {
            "kind": "training-throughput-tuning",
            "system": system,
            "seconds_per_trial": seconds_per_trial,
            "warmup_steps": warmup_steps,
            "trials": [asdict(trial) for trial in trials],
            "winner": asdict(winner.candidate),
        },
    )
    return winner.candidate


def parse_tuning_candidates(values: Sequence[str]) -> tuple[TuningCandidate, ...]:
    """Decode runtime-supplied worker:prefetch controls."""
    candidates = []
    for value in values:
        try:
            workers, prefetch = value.split(":", maxsplit=1)
            candidate = TuningCandidate(int(workers), int(prefetch))
        except ValueError as error:
            raise ValueError("tuning candidates use workers:prefetch syntax") from error
        candidate.validate()
        candidates.append(candidate)
    if len(set(candidates)) != len(candidates):
        raise ValueError("tuning candidates must be unique")
    return tuple(candidates)

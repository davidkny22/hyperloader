"""Cross-machine training legs and uninterrupted loader replay."""

from __future__ import annotations

from typing import Any, Protocol

from .hash_chain import EMPTY_HASH_CHAIN, advance_hash_chain
from .public_feeders import TrainingBatch
from .resume_records import ARITHMETIC_CONTRACT, ResumeLeg


class StatefulBatchSource(Protocol):
    """One resumable loader iterator and its delivered-coordinate owner."""

    worker_count: int

    def next_batch(self) -> TrainingBatch:
        """Return the next delivered batch."""

    def state_dict(self) -> dict[str, object]:
        """Capture the exact next-batch coordinate."""

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore the exact next-batch coordinate."""


class ResumeStepRunner(Protocol):
    """Model and optimizer state used by checkpointed training legs."""

    model: Any
    optimizer: Any

    def step(self, batch: TrainingBatch) -> Any:
        """Execute one training step."""

    def finish(self, loss: Any) -> float:
        """Settle device work and return the terminal scalar loss."""


def run_resume_leg(
    source: StatefulBatchSource,
    runner: ResumeStepRunner,
    *,
    ordinal: int,
    machine: str,
    steps: int,
    optimizer_step_start: int,
    initial_hash_chain: str = EMPTY_HASH_CHAIN,
    initial_loss: float | None = None,
) -> ResumeLeg:
    """Execute one fixed-step training leg and advance the delivered-batch chain."""
    if ordinal < 0 or not machine or steps <= 0 or optimizer_step_start < 0:
        raise ValueError("resume leg controls are invalid")
    chain = initial_hash_chain
    terminal: Any | None = None
    for _ in range(steps):
        batch = source.next_batch()
        batch.validate()
        terminal = runner.step(batch)
        chain = advance_hash_chain(chain, batch.digest)
    assert terminal is not None
    terminal_loss = runner.finish(terminal)
    return ResumeLeg(
        ordinal=ordinal,
        machine=machine,
        worker_count=source.worker_count,
        optimizer_step_start=optimizer_step_start,
        optimizer_step_stop=optimizer_step_start + steps,
        initial_hash_chain=initial_hash_chain,
        final_hash_chain=chain,
        initial_loss=initial_loss,
        terminal_loss=terminal_loss,
        arithmetic_contract=ARITHMETIC_CONTRACT,
    )


def replay_loader_hash_chain(
    source: StatefulBatchSource,
    *,
    batches: int,
    initial_hash_chain: str = EMPTY_HASH_CHAIN,
) -> str:
    """Compute the uninterrupted loader-only oracle without model execution."""
    if batches <= 0:
        raise ValueError("loader replay requires at least one batch")
    chain = initial_hash_chain
    for _ in range(batches):
        batch = source.next_batch()
        batch.validate()
        chain = advance_hash_chain(chain, batch.digest)
    return chain

"""Machine-readable records for three-leg training resume evidence."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .hash_chain import EMPTY_HASH_CHAIN
from .output import write_result

ARITHMETIC_CONTRACT = "continuous-not-bit-identical"


@dataclass(frozen=True)
class ResumeLeg:
    """One machine-local segment of a cross-machine training run."""

    ordinal: int
    machine: str
    worker_count: int
    optimizer_step_start: int
    optimizer_step_stop: int
    initial_hash_chain: str
    final_hash_chain: str
    initial_loss: float | None
    terminal_loss: float
    arithmetic_contract: str = ARITHMETIC_CONTRACT


@dataclass(frozen=True)
class CheckpointRecord:
    """Portable checkpoint artifact written after one nonterminal leg."""

    after_leg: int
    artifact_name: str
    sha256: str
    loader_state_sha256: str


@dataclass(frozen=True)
class ResumeBundle:
    """Three legs, two checkpoints, and an uninterrupted stream oracle."""

    evaluation_id: str
    run_id: str
    legs: tuple[ResumeLeg, ResumeLeg, ResumeLeg]
    checkpoints: tuple[CheckpointRecord, CheckpointRecord]
    oracle_hash_chain: str
    visual_artifacts: tuple[str, ...] = ()


def validate_resume_bundle(bundle: ResumeBundle) -> None:
    """Require exact stream continuity without claiming arithmetic identity."""
    if not bundle.evaluation_id or not bundle.run_id or bundle.visual_artifacts:
        raise ValueError("resume evidence requires identities and no visual artifacts")
    legs = bundle.legs
    if tuple(leg.ordinal for leg in legs) != (0, 1, 2):
        raise ValueError("resume legs must have ordinals zero, one, and two")
    if legs[0].machine != legs[2].machine or legs[1].machine == legs[0].machine:
        raise ValueError("resume topology must return to the first machine")
    if len({leg.worker_count for leg in legs}) < 2:
        raise ValueError("resume legs must exercise changed worker counts")
    previous_step = legs[0].optimizer_step_start
    previous_chain = EMPTY_HASH_CHAIN
    previous_loss: float | None = None
    for leg in legs:
        if leg.worker_count <= 0 or leg.optimizer_step_start != previous_step:
            raise ValueError("resume optimizer steps must be positive and contiguous")
        if leg.optimizer_step_stop <= leg.optimizer_step_start:
            raise ValueError("each resume leg must execute at least one optimizer step")
        if leg.initial_hash_chain != previous_chain:
            raise ValueError("resume batch hash chains are not contiguous")
        if leg.initial_loss != previous_loss:
            raise ValueError("resume loss records are not continuous across checkpoints")
        if not math.isfinite(leg.terminal_loss):
            raise ValueError("resume losses must be finite")
        if leg.arithmetic_contract != ARITHMETIC_CONTRACT:
            raise ValueError("resume arithmetic must be declared non-bit-identical")
        previous_step = leg.optimizer_step_stop
        previous_chain = leg.final_hash_chain
        previous_loss = leg.terminal_loss
    if previous_chain != bundle.oracle_hash_chain:
        raise ValueError("resumed batch chain does not match the uninterrupted oracle")
    if tuple(record.after_leg for record in bundle.checkpoints) != (0, 1):
        raise ValueError("resume evidence requires checkpoints after the first two legs")
    for record in bundle.checkpoints:
        if not record.artifact_name or not _is_sha256(record.sha256):
            raise ValueError("checkpoint artifact identity is invalid")
        if not _is_sha256(record.loader_state_sha256):
            raise ValueError("checkpoint loader-state identity is invalid")


def write_resume_bundle(path: Path, bundle: ResumeBundle) -> None:
    """Validate and write one canonical JSON-only resume evidence bundle."""
    validate_resume_bundle(bundle)
    write_result(path, {"kind": "cross-machine-resume", "resume": asdict(bundle)})


def _is_sha256(value: str) -> bool:
    try:
        return len(bytes.fromhex(value)) == 32
    except ValueError:
        return False

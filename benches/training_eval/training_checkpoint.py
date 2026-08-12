"""Portable model, optimizer, and loader checkpoints for resume legs."""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .resume_records import CheckpointRecord


@dataclass(frozen=True)
class ResumeCursor:
    """Training coordinate restored with a portable checkpoint."""

    optimizer_step: int
    batch_hash_chain: str
    terminal_loss: float


def save_training_checkpoint(
    path: Path,
    source: Any,
    runner: Any,
    *,
    after_leg: int,
    optimizer_step: int,
    batch_hash_chain: str,
    terminal_loss: float,
) -> CheckpointRecord:
    """Atomically save portable training and exact loader coordinate state."""
    loader_state = source.state_dict()
    loader_digest = _loader_state_digest(loader_state)
    payload = {
        "format": "hyperloader-training-checkpoint-1",
        "model": runner.model.state_dict(),
        "optimizer": runner.optimizer.state_dict(),
        "loader": loader_state,
        "optimizer_step": optimizer_step,
        "batch_hash_chain": batch_hash_chain,
        "terminal_loss": terminal_loss,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return CheckpointRecord(
        after_leg=after_leg,
        artifact_name=path.name,
        sha256=_file_digest(path),
        loader_state_sha256=loader_digest,
    )


def load_training_checkpoint(
    path: Path,
    source: Any,
    runner: Any,
    *,
    expected_sha256: str,
    map_location: str | torch.device,
) -> ResumeCursor:
    """Verify and restore a checkpoint before the next machine leg starts."""
    if _file_digest(path) != expected_sha256:
        raise ValueError("training checkpoint digest does not match its record")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != "hyperloader-training-checkpoint-1":
        raise ValueError("training checkpoint format is unsupported")
    runner.model.load_state_dict(payload["model"])
    runner.optimizer.load_state_dict(payload["optimizer"])
    source.load_state_dict(payload["loader"])
    return ResumeCursor(
        optimizer_step=int(payload["optimizer_step"]),
        batch_hash_chain=str(payload["batch_hash_chain"]),
        terminal_loss=float(payload["terminal_loss"]),
    )


def _loader_state_digest(state: dict[str, object]) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

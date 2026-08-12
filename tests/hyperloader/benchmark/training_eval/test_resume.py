"""Three-leg checkpoint and loader-oracle behavior tests."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from benches.training_eval import (
    ResumeBundle,
    TokenBatch,
    load_training_checkpoint,
    replay_loader_hash_chain,
    run_resume_leg,
    save_training_checkpoint,
    validate_resume_bundle,
    write_resume_bundle,
)


def _batch(index: int) -> TokenBatch:
    tokens = torch.full((2, 3), index + 1, dtype=torch.int64)
    return TokenBatch(tokens, hashlib.sha256(tokens.numpy().tobytes()).hexdigest())


class _Source:
    def __init__(self, worker_count: int) -> None:
        self.worker_count = worker_count
        self.cursor = 0
        self.batches = tuple(_batch(index) for index in range(6))

    def next_batch(self) -> TokenBatch:
        batch = self.batches[self.cursor]
        self.cursor += 1
        return batch

    def state_dict(self) -> dict[str, object]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.cursor = int(state["cursor"])


class _Runner:
    def __init__(self) -> None:
        self.model = torch.nn.Linear(3, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)

    def step(self, batch: TokenBatch) -> torch.Tensor:
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.model(batch.tokens.float()).square().mean()
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def finish(self, loss: torch.Tensor) -> float:
        return float(loss.item())


def test_three_leg_resume_matches_uninterrupted_loader_oracle(tmp_path: Path) -> None:
    torch.manual_seed(17)
    first_source, first_runner = _Source(2), _Runner()
    first = run_resume_leg(
        first_source,
        first_runner,
        ordinal=0,
        machine="spark",
        steps=2,
        optimizer_step_start=0,
    )
    first_record = save_training_checkpoint(
        tmp_path / "first.pt",
        first_source,
        first_runner,
        after_leg=0,
        optimizer_step=first.optimizer_step_stop,
        batch_hash_chain=first.final_hash_chain,
        terminal_loss=first.terminal_loss,
    )

    second_source, second_runner = _Source(4), _Runner()
    second_cursor = load_training_checkpoint(
        tmp_path / "first.pt",
        second_source,
        second_runner,
        expected_sha256=first_record.sha256,
        map_location="cpu",
    )
    second = run_resume_leg(
        second_source,
        second_runner,
        ordinal=1,
        machine="rtx-4070",
        steps=2,
        optimizer_step_start=second_cursor.optimizer_step,
        initial_hash_chain=second_cursor.batch_hash_chain,
        initial_loss=second_cursor.terminal_loss,
    )
    second_record = save_training_checkpoint(
        tmp_path / "second.pt",
        second_source,
        second_runner,
        after_leg=1,
        optimizer_step=second.optimizer_step_stop,
        batch_hash_chain=second.final_hash_chain,
        terminal_loss=second.terminal_loss,
    )

    third_source, third_runner = _Source(1), _Runner()
    third_cursor = load_training_checkpoint(
        tmp_path / "second.pt",
        third_source,
        third_runner,
        expected_sha256=second_record.sha256,
        map_location="cpu",
    )
    third = run_resume_leg(
        third_source,
        third_runner,
        ordinal=2,
        machine="spark",
        steps=2,
        optimizer_step_start=third_cursor.optimizer_step,
        initial_hash_chain=third_cursor.batch_hash_chain,
        initial_loss=third_cursor.terminal_loss,
    )
    oracle = replay_loader_hash_chain(_Source(3), batches=6)
    bundle = ResumeBundle(
        "resume-eval",
        "run-1",
        (first, second, third),
        (first_record, second_record),
        oracle,
    )
    validate_resume_bundle(bundle)
    output = tmp_path / "resume.json"
    write_resume_bundle(output, bundle)
    assert json.loads(output.read_text(encoding="utf-8"))["resume"]["oracle_hash_chain"] == oracle
    assert all(math.isfinite(leg.terminal_loss) for leg in bundle.legs)


def test_resume_bundle_rejects_a_stream_divergence(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    changed = dataclasses.replace(bundle, oracle_hash_chain="f" * 64)
    with pytest.raises(ValueError, match="oracle"):
        validate_resume_bundle(changed)


def test_checkpoint_digest_detects_transfer_corruption(tmp_path: Path) -> None:
    source, runner = _Source(2), _Runner()
    leg = run_resume_leg(
        source,
        runner,
        ordinal=0,
        machine="spark",
        steps=1,
        optimizer_step_start=0,
    )
    record = save_training_checkpoint(
        tmp_path / "checkpoint.pt",
        source,
        runner,
        after_leg=0,
        optimizer_step=leg.optimizer_step_stop,
        batch_hash_chain=leg.final_hash_chain,
        terminal_loss=leg.terminal_loss,
    )
    with (tmp_path / "checkpoint.pt").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        load_training_checkpoint(
            tmp_path / "checkpoint.pt",
            _Source(3),
            _Runner(),
            expected_sha256=record.sha256,
            map_location="cpu",
        )


def _valid_bundle(tmp_path: Path) -> ResumeBundle:
    first = run_resume_leg(_Source(2), _Runner(), ordinal=0, machine="spark", steps=1, optimizer_step_start=0)
    second = dataclasses.replace(
        first,
        ordinal=1,
        machine="rtx-4070",
        worker_count=4,
        optimizer_step_start=1,
        optimizer_step_stop=2,
        initial_hash_chain=first.final_hash_chain,
        final_hash_chain="a" * 64,
        initial_loss=first.terminal_loss,
    )
    third = dataclasses.replace(
        second,
        ordinal=2,
        machine="spark",
        worker_count=1,
        optimizer_step_start=2,
        optimizer_step_stop=3,
        initial_hash_chain=second.final_hash_chain,
        final_hash_chain="b" * 64,
        initial_loss=second.terminal_loss,
    )
    digest = hashlib.sha256(b"checkpoint").hexdigest()
    from benches.training_eval import CheckpointRecord

    records = (
        CheckpointRecord(0, "first.pt", digest, digest),
        CheckpointRecord(1, "second.pt", digest, digest),
    )
    return ResumeBundle("eval", "run", (first, second, third), records, third.final_hash_chain)

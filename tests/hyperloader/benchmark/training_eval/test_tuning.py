"""Behavior tests for counted live feeder tuning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from benches.training_eval.dial import TransformerDialPoint
from benches.training_eval.token_source import PretokenizedRows
from benches.training_eval.token_tuning import tune_token_system
from benches.training_eval.transformer import DialTransformer
from benches.training_eval.tuning import (
    TuningCandidate,
    parse_tuning_candidates,
    tune_live_feeder,
)


@dataclass(frozen=True)
class _Batch:
    samples: int = 4


class _Feeder:
    system = "subject"

    def __init__(self) -> None:
        self.closed = False

    def next_batch(self) -> _Batch:
        return _Batch()

    def close(self) -> None:
        self.closed = True


class _Runner:
    def step(self, batch: _Batch) -> float:
        return float(batch.samples)

    def finish(self, loss: float) -> float:
        return loss


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.25
        return self.value


def test_tuner_records_equal_candidates_and_closes_feeders(tmp_path: Path) -> None:
    feeders = []

    def build(candidate: TuningCandidate):
        feeder = _Feeder()
        feeders.append((candidate, feeder))
        return feeder, _Runner()

    output = tmp_path / "tuning.json"
    selected = tune_live_feeder(
        "subject",
        (TuningCandidate(1, 2), TuningCandidate(2, 2)),
        build_trial=build,
        seconds_per_trial=0.5,
        warmup_steps=1,
        output=output,
        clock=_Clock(),
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    assert selected in {TuningCandidate(1, 2), TuningCandidate(2, 2)}
    assert len(document["trials"]) == 2
    assert all(feeder.closed for _, feeder in feeders)


def test_runtime_candidate_parser_rejects_duplicates_and_bad_syntax() -> None:
    assert parse_tuning_candidates(("2:4",)) == (TuningCandidate(2, 4),)
    for values in (("2:4", "2:4"), ("bad",), ("0:2",)):
        try:
            parse_tuning_candidates(values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid tuning controls accepted: {values}")


def test_token_tuning_executes_real_public_feeder_and_optimizer(tmp_path: Path) -> None:
    point = TransformerDialPoint("tiny", 8, 1, 2, 4, 2, 17)
    selected = tune_token_system(
        "torch",
        dataset=PretokenizedRows(rows=4, sequence_length=4, vocabulary_size=17, seed=5),
        batch_size=2,
        model_factory=lambda: DialTransformer(point),
        candidates=(TuningCandidate(1, 1),),
        device=torch.device("cpu"),
        precision="float32",
        learning_rate=0.0003,
        pin_memory=False,
        seed=5,
        seconds_per_trial=0.001,
        warmup_steps=1,
        output=tmp_path / "tuning.json",
    )
    assert selected == TuningCandidate(1, 1)

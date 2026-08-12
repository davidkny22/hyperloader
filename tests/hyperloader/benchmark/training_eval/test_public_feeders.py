"""Public training-feeder behavior tests."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from benches.training_eval import build_public_feeder, collate_token_batch


class _TokenRows(Dataset[torch.Tensor]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor((index, index + 1, index + 2), dtype=torch.int64)


def test_torch_and_hyperloader_public_paths_deliver_identical_batches() -> None:
    expected = _collect("torch", workers=2)
    actual = _collect("hyperloader", workers=2)
    assert actual == expected


def test_spdl_public_path_delivers_the_same_ordered_batches() -> None:
    assert _collect("spdl", workers=2) == _collect("torch", workers=0)


def test_public_feeder_cycles_at_complete_epoch_boundaries() -> None:
    feeder = build_public_feeder(
        "torch",
        _TokenRows(),
        batch_size=2,
        workers=0,
        prefetch=2,
        collate=collate_token_batch,
    )
    try:
        first = feeder.next_batch()
        feeder.next_batch()
        repeated = feeder.next_batch()
        assert repeated.digest == first.digest
        assert torch.equal(repeated.tokens, first.tokens)
    finally:
        feeder.close()


def test_hyperloader_public_feeder_restores_with_a_changed_worker_count() -> None:
    expected = _collect("torch", workers=0)
    source = build_public_feeder(
        "hyperloader",
        _TokenRows(),
        batch_size=2,
        workers=2,
        prefetch=2,
        collate=collate_token_batch,
    )
    try:
        first = source.next_batch()
        state = source.state_dict()
    finally:
        source.close()
    resumed = build_public_feeder(
        "hyperloader",
        _TokenRows(),
        batch_size=2,
        workers=1,
        prefetch=2,
        collate=collate_token_batch,
    )
    try:
        resumed.load_state_dict(state)
        second = resumed.next_batch()
    finally:
        resumed.close()
    assert (first.tokens.tolist(), first.digest) == expected[0]
    assert (second.tokens.tolist(), second.digest) == expected[1]


def _collect(system: str, *, workers: int) -> list[tuple[list[list[int]], str]]:
    feeder = build_public_feeder(
        system,
        _TokenRows(),
        batch_size=2,
        workers=workers,
        prefetch=2,
        collate=collate_token_batch,
    )
    try:
        return [
            (batch.tokens.tolist(), batch.digest)
            for batch in (feeder.next_batch(), feeder.next_batch())
        ]
    finally:
        feeder.close()

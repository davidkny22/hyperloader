"""Public training-feeder behavior tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset, TensorDataset

from benches.training_eval import build_public_feeder, collate_token_batch
from benches.training_eval.feeders import TokenViewAdapter


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


@pytest.mark.parametrize("system", ["torch", "hyperloader"])
def test_process_feeder_records_worker_boot_controls(
    system: str, tmp_path: Path
) -> None:
    feeder = build_public_feeder(
        system,
        _TokenRows(),
        batch_size=2,
        workers=2,
        prefetch=2,
        collate=collate_token_batch,
        worker_environment_dir=tmp_path,
    )
    try:
        feeder.next_batch()
        snapshot = feeder.control_snapshot()
    finally:
        feeder.close()
    assert len(snapshot["processes"]) == 2
    assert {record["worker_id"] for record in snapshot["worker_boot"]} == {0, 1}
    assert {
        record["torch_intra_op_threads"] for record in snapshot["worker_boot"]
    } == {1}


def test_hyperloader_public_feeder_delivers_tensor_dataset_views_without_workers() -> None:
    source = torch.arange(24, dtype=torch.int64).reshape(4, 6)
    first = collate_token_batch([source[0], source[1]])
    second = collate_token_batch([source[2], source[3]])
    feeder = build_public_feeder(
        "hyperloader",
        TensorDataset(source),
        batch_size=2,
        workers=2,
        prefetch=2,
        collate=None,
        pin_memory=torch.cuda.is_available(),
        batch_adapter=TokenViewAdapter(2, 6, (first.digest, second.digest)),
    )
    try:
        actual = feeder.next_batch()
        snapshot = feeder.control_snapshot()
        stats = feeder._loader.stats()
    finally:
        feeder.close()
    assert actual.digest == first.digest
    assert torch.equal(actual.tokens, first.tokens)
    assert actual.tokens.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    if torch.cuda.is_available():
        assert stats["memory"]["pinned_registered_bytes"] == source.nbytes
        assert stats["memory"]["pinned_staged_bytes"] == 0
    assert snapshot["configured_workers"] == 2
    assert snapshot["processes"] == []
    assert snapshot["worker_boot"] == []


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

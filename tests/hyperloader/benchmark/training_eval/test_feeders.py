"""Live-training feeder adapter behavior tests."""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import pytest
import torch

from benches.training_eval import IteratorTokenFeeder, ResidentTokenFeeder, TokenBatch


def _batch(value: int) -> TokenBatch:
    return TokenBatch(
        torch.full((2, 4), value, dtype=torch.int64),
        hashlib.sha256(bytes([value])).hexdigest(),
    )


def test_resident_feeder_cycles_prebuilt_batch_identity() -> None:
    first, second = _batch(1), _batch(2)
    feeder = ResidentTokenFeeder("counterfactual", (first, second))
    assert feeder.next_batch() is first
    assert feeder.next_batch() is second
    assert feeder.next_batch() is first


def test_iterator_feeder_does_not_hide_loader_exhaustion() -> None:
    batch = _batch(3)
    feeder = IteratorTokenFeeder("hyperloader", iter((batch,)))
    assert feeder.next_batch() is batch
    with pytest.raises(StopIteration):
        feeder.next_batch()


def test_token_batch_rejects_non_digest_metadata() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        TokenBatch(torch.ones((1, 2), dtype=torch.int64), "not-a-digest").validate()
    with pytest.raises(ValueError, match="SHA-256"):
        TokenBatch(torch.ones((1, 2), dtype=torch.int64), " " * 64).validate()


def test_token_batch_exposes_the_public_pinning_protocol() -> None:
    batch = _batch(3)
    with patch.object(torch.Tensor, "pin_memory", return_value=batch.tokens) as pin:
        pinned = batch.pin_memory()
    assert pinned.digest == batch.digest
    pin.assert_called_once_with()

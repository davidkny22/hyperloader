"""Worker-side homogeneous NumPy batching checks."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np
import torch

from hyperloader.process.batching import ArrayBatcher, unwrap_batch_payload
from hyperloader.process.serialization import ResultDecoder, ResultEncoder


@dataclass(frozen=True)
class Dispatch:
    """Minimal batch-boundary view exposed by the native worker command."""

    position: int
    batch_end: bool


class ArrayBatcherTest(unittest.TestCase):
    """Verify fusion, probe retention, fallback, and exception position."""

    def test_homogeneous_array_batch_publishes_ack_then_one_tensor(self) -> None:
        batcher = ArrayBatcher(2, ResultEncoder())

        self.assertEqual(
            batcher.success(Dispatch(0, False), np.arange(4, dtype=np.int64)),
            [],
        )
        completions = batcher.success(
            Dispatch(1, True), np.arange(4, 8, dtype=np.int64)
        )

        self.assertEqual(len(completions), 2)
        self.assertEqual(completions[0].payload, b"")
        payload = unwrap_batch_payload(completions[1].payload)
        self.assertIsNotNone(payload)
        batch = ResultDecoder().decode(payload, worker=0)
        self.assertTrue(
            torch.equal(batch, torch.arange(8, dtype=torch.int64).reshape(2, 4))
        )

    def test_probe_value_is_materialized_without_a_second_execution(self) -> None:
        batcher = ArrayBatcher(2, ResultEncoder())
        batcher.seed_probe(np.arange(4, dtype=np.int64))

        completions = batcher.success(
            Dispatch(1, True), np.arange(4, 8, dtype=np.int64)
        )

        self.assertEqual(len(completions), 1)
        payload = unwrap_batch_payload(completions[0].payload)
        self.assertIsNotNone(payload)
        batch = ResultDecoder().decode(payload, worker=0)
        self.assertEqual(batch.shape, (2, 4))

    def test_incompatible_array_flushes_ordinary_sample_payloads(self) -> None:
        batcher = ArrayBatcher(2, ResultEncoder())
        batcher.success(Dispatch(0, False), np.arange(4, dtype=np.int64))

        completions = batcher.success(
            Dispatch(1, True), np.arange(6, dtype=np.int64)
        )

        self.assertEqual(len(completions), 2)
        self.assertTrue(all(completion.payload for completion in completions))
        decoder = ResultDecoder()
        self.assertEqual(decoder.decode(completions[0].payload, 0).shape, (4,))
        self.assertEqual(decoder.decode(completions[1].payload, 0).shape, (6,))

    def test_failure_flushes_prior_value_and_retains_failing_dispatch(self) -> None:
        batcher = ArrayBatcher(2, ResultEncoder())
        batcher.success(Dispatch(0, False), np.arange(4, dtype=np.int64))

        completions = batcher.failure(Dispatch(1, True), 1, b"failure")

        self.assertEqual([item.dispatch.position for item in completions], [0, 1])
        self.assertEqual([item.status for item in completions], [0, 1])
        self.assertEqual(completions[1].payload, b"failure")


if __name__ == "__main__":
    unittest.main()

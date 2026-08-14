"""One-ahead staging-thread delivery checks."""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from hyperloader import DataLoader, diagnose
from hyperloader.control import StagedCopyTax

from hyperloader.memory.pinned.iterator import PinnedDeliveryIterator


class _Delivery:
    def __init__(self) -> None:
        self.consumer_threads: set[int] = set()
        self.staging_threads: set[int] = set()

    def bind_consumer_thread(self, thread_id: int) -> None:
        self.consumer_threads.add(thread_id)

    def stage(self, value: int) -> int:
        self.staging_threads.add(threading.get_ident())
        return value * 10


class _Iterator:
    def __init__(self) -> None:
        self.values = iter((1, 2, 3))
        self.calls = 0
        self.invalid = False

    def __next__(self) -> int:
        self.calls += 1
        return next(self.values)

    @property
    def complete(self) -> bool:
        return False

    @property
    def coordinate_epoch(self) -> int:
        return 4

    @property
    def delivered_batches(self) -> int:
        return self.calls

    @property
    def sampler_checksum(self) -> int:
        return self.calls * 7

    @property
    def delivered_bitmap(self) -> bytes:
        return bytes((self.calls,))

    def invalidate(self) -> None:
        self.invalid = True


class _TensorRows:
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor([index, index + 1])


class PinnedDeliveryIteratorTest(unittest.TestCase):
    """Prove thread ownership, one-ahead depth, and delivered-only state."""

    def test_stage_runs_off_consumer_and_prefetches_exactly_one_batch(self) -> None:
        delivery = _Delivery()
        inner = _Iterator()
        iterator = PinnedDeliveryIterator(delivery, inner)
        consumer = threading.get_ident()

        self.assertEqual(next(iterator), 10)
        deadline = time.monotonic() + 1.0
        while inner.calls < 2 and time.monotonic() < deadline:
            time.sleep(0.001)

        self.assertEqual(inner.calls, 2)
        self.assertEqual(iterator.delivered_batches, 1)
        self.assertEqual(iterator.sampler_checksum, 7)
        self.assertEqual(iterator.delivered_bitmap, b"\x01")
        self.assertNotIn(consumer, delivery.staging_threads)
        self.assertEqual(len(delivery.staging_threads), 1)

        time.sleep(0.01)
        self.assertEqual(inner.calls, 2)
        self.assertEqual(next(iterator), 20)
        iterator.close()
        self.assertTrue(inner.invalid)

    def test_exhaustion_is_forwarded_after_the_last_staged_batch(self) -> None:
        iterator = PinnedDeliveryIterator(_Delivery(), _Iterator())

        self.assertEqual([next(iterator), next(iterator), next(iterator)], [10, 20, 30])
        with self.assertRaises(StopIteration):
            next(iterator)

        self.assertTrue(iterator.complete)
        iterator.close()

    def test_public_auto_path_reports_dedicated_one_ahead_staging(self) -> None:
        loader = DataLoader(
            _TensorRows(),
            batch_size=2,
            num_workers=1,
            prefetch_factor=1,
            seed=41,
        )
        loader._calibration = SimpleNamespace(
            staged_copy_tax=StagedCopyTax(16, 10, 20),
            idle_state_tax=None,
        )
        empty_strided = torch.empty_strided
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch(
                "torch.empty_strided",
                side_effect=lambda shape, stride, **options: empty_strided(
                    shape, stride, dtype=options["dtype"]
                ),
            ),
        ):
            iterator = iter(loader)
            try:
                batch = next(iterator)
                report = diagnose(loader).record["telemetry"]["memory"]
            finally:
                loader.close()

        self.assertTrue(torch.equal(batch, torch.tensor([[0, 1], [1, 2]])))
        self.assertEqual(report["delivery_memory"], "pinned")
        self.assertEqual(report["staging_prefetch_depth"], 1)
        self.assertEqual(report["staging_thread_count"], 1)
        self.assertFalse(report["staging_on_consumer_thread"])

    def test_public_auto_path_delivers_host_when_staging_loses(self) -> None:
        loader = DataLoader(_TensorRows(), batch_size=2, num_workers=0, seed=43)
        loader._calibration = SimpleNamespace(
            staged_copy_tax=StagedCopyTax(16, 20, 10),
            idle_state_tax=None,
        )
        with mock.patch("torch.cuda.is_available", return_value=True):
            iterator = iter(loader)
            try:
                batch = next(iterator)
                report = diagnose(loader).record["telemetry"]["memory"]
            finally:
                loader.close()

        self.assertTrue(torch.equal(batch, torch.tensor([[0, 1], [1, 2]])))
        self.assertEqual(report["delivery_memory"], "host")
        self.assertEqual(report["staging_prefetch_depth"], 0)
        self.assertEqual(report["staging_thread_count"], 0)


if __name__ == "__main__":
    unittest.main()

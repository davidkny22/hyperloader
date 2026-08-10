"""Declared thread-tier routing and parity tests."""

from __future__ import annotations

import threading
import unittest
from unittest import mock

import torch
from hyperloader import DataLoader, rng
from hyperloader.thread import ThreadPool


class PortableDataset:
    """Use only sanctioned per-sample state from user code."""

    def __len__(self) -> int:
        return 12

    def __getitem__(self, index: int) -> tuple[int, float, float, float]:
        return (
            index,
            float(torch.rand((), generator=rng()).item()),
            float(rng("numpy").random()),
            rng("random").random(),
        )


class ThreadIteratorTest(unittest.TestCase):
    """Prove routing, declaration boundaries, and process identity."""

    def test_declared_dataset_is_bit_equal_across_process_and_thread(self) -> None:
        process = DataLoader(
            PortableDataset(), batch_size=None, num_workers=2, seed=41
        )
        threaded = DataLoader(
            PortableDataset(),
            batch_size=None,
            num_workers=2,
            seed=41,
            thread_safe=True,
        )
        try:
            self.assertEqual(list(threaded), list(process))
        finally:
            threaded.close()
            process.close()

    def test_declaration_requires_a_boolean(self) -> None:
        with self.assertRaisesRegex(TypeError, "boolean declaration"):
            DataLoader(PortableDataset(), num_workers=1, thread_safe="yes")

    def test_thread_safe_declaration_routes_through_thread_pool(self) -> None:
        loader = DataLoader(
            PortableDataset(),
            batch_size=None,
            num_workers=2,
            seed=43,
            thread_safe=True,
        )
        original = ThreadPool.submit
        threads: set[int] = set()

        def recording_submit(pool: ThreadPool, *args: object) -> object:
            threads.add(threading.get_ident())
            return original(pool, *args)

        try:
            with mock.patch.object(ThreadPool, "submit", recording_submit):
                self.assertEqual(len(list(loader)), len(PortableDataset()))
            self.assertTrue(threads)
        finally:
            loader.close()

    def test_severed_thread_submit_breaks_the_public_path(self) -> None:
        loader = DataLoader(
            PortableDataset(),
            batch_size=None,
            num_workers=1,
            seed=44,
            thread_safe=True,
        )
        try:
            with (
                mock.patch.object(
                    ThreadPool, "submit", side_effect=RuntimeError("severed thread path")
                ),
                self.assertRaisesRegex(RuntimeError, "severed thread path"),
            ):
                iter(loader)
        finally:
            loader.close()

    def test_worker_initializer_cannot_access_stage_rng(self) -> None:
        def initialize(_worker_id: int) -> None:
            rng()

        loader = DataLoader(
            PortableDataset(),
            batch_size=None,
            num_workers=1,
            seed=47,
            thread_safe=True,
            worker_init_fn=initialize,
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "only while user code"):
                next(iter(loader))
        finally:
            loader.close()

    def test_tensor_plan_keeps_storage_identity_route(self) -> None:
        dataset = torch.arange(16)
        loader = DataLoader(
            dataset, batch_size=4, num_workers=2, thread_safe=True, seed=53
        )
        try:
            batch = next(iter(loader))
            self.assertEqual(batch.untyped_storage().data_ptr(), dataset.data_ptr())
            self.assertIsNone(loader._thread_pool)
        finally:
            loader.close()


if __name__ == "__main__":
    unittest.main()

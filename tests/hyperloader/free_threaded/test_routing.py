"""Installed free-threaded execution routing assurance."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from hyperloader import DataLoader
from hyperloader.thread import ThreadPool
from hyperloader.thread.gil import free_threaded_build, gil_enabled


class PurePythonDataset:
    """A picklable transform whose execution identity is observable."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> tuple[int, int]:
        return index, index * index


class FreeThreadedRoutingGate(unittest.TestCase):
    """Prove build detection and declaration-gated public routing."""

    def test_runtime_is_a_free_threaded_build_with_gil_disabled(self) -> None:
        self.assertTrue(free_threaded_build())
        self.assertIs(gil_enabled(), False)

    def test_undeclared_dataset_keeps_process_execution(self) -> None:
        sever_route = os.environ.get("HYPERLOADER_TEST_SEVER_PROCESS_ROUTE") == "1"
        if sever_route:
            with mock.patch("hyperloader.api.prepare_process_pool"):
                loader = DataLoader(
                    PurePythonDataset(), batch_size=None, num_workers=2, seed=71
                )
        else:
            loader = DataLoader(
                PurePythonDataset(), batch_size=None, num_workers=2, seed=71
            )
        try:
            self.assertIsNotNone(loader._process_pool)
            self.assertIsNone(loader._thread_pool)
            self.assertEqual(list(loader), [(0, 0), (1, 1), (2, 4), (3, 9)])
        finally:
            loader.close()

    def test_declared_pure_python_dataset_uses_thread_execution(self) -> None:
        loader = DataLoader(
            PurePythonDataset(),
            batch_size=None,
            num_workers=2,
            seed=73,
            thread_safe=True,
        )
        try:
            self.assertIsNone(loader._process_pool)
            self.assertEqual(list(loader), [(0, 0), (1, 1), (2, 4), (3, 9)])
            self.assertIsInstance(loader._thread_pool, ThreadPool)
        finally:
            loader.close()

    def test_non_free_threaded_build_identity_is_rejected(self) -> None:
        with mock.patch(
            "hyperloader.thread.gil.sysconfig.get_config_var", return_value=0
        ):
            self.assertFalse(free_threaded_build())


if __name__ == "__main__":
    unittest.main()

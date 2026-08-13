"""Installed public-path process-worker environment behavior."""

from __future__ import annotations

import unittest

import torch
from hyperloader import DataLoader


class ThreadReportingDataset:
    """Report the Torch thread count seen while loading and evaluating."""

    def __init__(self) -> None:
        self._load_threads = -1

    def __getstate__(self) -> dict[str, int]:
        return {}

    def __setstate__(self, _state: dict[str, int]) -> None:
        self._load_threads = torch.get_num_threads()

    def __len__(self) -> int:
        return 2

    def __getitem__(self, _index: int) -> tuple[int, int]:
        return self._load_threads, torch.get_num_threads()


def raise_worker_thread_count(_worker_id: int) -> None:
    """Exercise the documented post-boot user override."""
    torch.set_num_threads(2)


class WorkerEnvironmentTest(unittest.TestCase):
    def test_dataset_load_and_execution_see_one_intra_op_thread(self) -> None:
        loader = DataLoader(ThreadReportingDataset(), batch_size=2, num_workers=1)
        try:
            loaded, executed = next(iter(loader))
        finally:
            loader.close()

        self.assertEqual(loaded.tolist(), [1, 1])
        self.assertEqual(executed.tolist(), [1, 1])

    def test_worker_init_can_raise_thread_count_after_dataset_load(self) -> None:
        loader = DataLoader(
            ThreadReportingDataset(),
            batch_size=2,
            num_workers=1,
            worker_init_fn=raise_worker_thread_count,
        )
        try:
            loaded, executed = next(iter(loader))
        finally:
            loader.close()

        self.assertEqual(loaded.tolist(), [1, 1])
        self.assertEqual(executed.tolist(), [2, 2])


if __name__ == "__main__":
    unittest.main()

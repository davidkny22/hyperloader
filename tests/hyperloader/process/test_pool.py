"""Persistent process execution through the installed public package."""

from __future__ import annotations

import os
import random
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.utils.data import get_worker_info

from hyperloader import DataLoader, _hyperloader
from hyperloader.process import ProcessPool


class RandomDataset:
    """Return process identity and draws from every seeded global."""

    def __init__(self, length: int = 4) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, object]:
        info = get_worker_info()
        return {
            "index": index,
            "pid": os.getpid(),
            "worker": info.id,
            "seed": info.seed,
            "torch": torch.rand(()).item(),
            "python": random.random(),
            "numpy": np.random.random(),
        }


class FailingDataset:
    """Raise a user exception away from the construction probe."""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> int:
        if index == 1:
            raise ValueError("sample failed")
        return index


class PublicDataset:
    """Return values supported by the engine's int64 collation contract."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "index": index,
            "pid": os.getpid(),
            "torch": torch.rand(()).item(),
        }


class DelayedDataset:
    """Record worker execution order while delaying one frontier head."""

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> int:
        if index == 1:
            time.sleep(0.2)
        with Path(self.log_path).open("a", encoding="utf-8") as stream:
            stream.write(f"{index}\n")
        return index


def record_worker_init(_worker_id: int) -> None:
    """Record the once-per-process initialization view."""
    info = get_worker_info()
    directory = Path(os.environ["HYPERLOADER_INIT_LOG"])
    (directory / f"worker-{info.id}.log").write_text(
        f"{os.getpid()}:{info.seed}\n", encoding="utf-8"
    )


def random_signature(value: dict[str, object]) -> tuple[object, ...]:
    """Remove execution identity from a random sample result."""
    return (
        value["index"],
        value["seed"],
        value["torch"],
        value["python"],
        value["numpy"],
    )


class ProcessPoolTest(unittest.TestCase):
    """Exercise persistence, RNG installation, errors, and public delivery."""

    def test_per_sample_rng_reproduces_across_fresh_pools(self) -> None:
        first = ProcessPool(RandomDataset(), 2, 17, 0, 0, 0)
        second = ProcessPool(RandomDataset(), 2, 17, 0, 0, 0)
        try:
            first_values = [first.execute(0, position, position) for position in range(4)]
            second_values = [
                second.execute(0, position, position) for position in range(4)
            ]
        finally:
            first.close()
            second.close()

        self.assertEqual(
            [random_signature(value) for value in first_values],
            [random_signature(value) for value in second_values],
        )
        self.assertEqual(len({value["seed"] for value in first_values}), 4)
        for position, value in enumerate(first_values):
            expected_seed = _hyperloader._sample_seed_words(17, 0, position)[0]
            self.assertEqual(value["seed"], expected_seed)

    def test_worker_init_runs_once_with_no_sample_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            previous = os.environ.get("HYPERLOADER_INIT_LOG")
            os.environ["HYPERLOADER_INIT_LOG"] = directory
            pool = ProcessPool(
                RandomDataset(),
                2,
                23,
                0,
                0,
                0,
                worker_init_fn=record_worker_init,
            )
            try:
                for position in range(4):
                    pool.execute(0, position, position)
                expected_pids = set(pool.worker_pids)
            finally:
                pool.close()
                if previous is None:
                    os.environ.pop("HYPERLOADER_INIT_LOG", None)
                else:
                    os.environ["HYPERLOADER_INIT_LOG"] = previous

            records = [
                path.read_text(encoding="utf-8").strip()
                for path in sorted(Path(directory).glob("worker-*.log"))
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual({int(row.split(":")[0]) for row in records}, expected_pids)
            self.assertTrue(all(row.endswith(":None") for row in records))

    def test_user_exception_keeps_formatted_worker_traceback(self) -> None:
        pool = ProcessPool(FailingDataset(), 1, 29, 0, 0, 0)
        try:
            with self.assertRaisesRegex(ValueError, "Original traceback") as raised:
                pool.execute(0, 1, 1)
        finally:
            pool.close()

        self.assertIn("sample failed", str(raised.exception))
        self.assertIn("__getitem__", str(raised.exception))

    def test_public_loader_reuses_workers_across_epochs(self) -> None:
        loader = DataLoader(PublicDataset(), batch_size=2, num_workers=2, seed=31)
        try:
            first = list(loader)
            pids = set(loader._process_pool.worker_pids)
            second = list(loader)
        finally:
            loader.close()

        self.assertEqual(sum(batch["index"].numel() for batch in first), 4)
        self.assertEqual(
            {int(pid) for batch in first + second for pid in batch["pid"]}, pids
        )
        self.assertFalse(torch.equal(first[0]["torch"], second[0]["torch"]))

    def test_public_loader_reorders_real_out_of_order_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "execution-order.log"
            loader = DataLoader(
                DelayedDataset(str(log_path)), batch_size=1, num_workers=2, seed=37
            )
            try:
                delivered = [int(batch.item()) for batch in loader]
            finally:
                loader.close()

            executed = [int(value) for value in log_path.read_text().splitlines()]

        self.assertEqual(delivered, [0, 1, 2, 3])
        self.assertLess(executed.index(2), executed.index(1))
        self.assertEqual(executed.count(0), 1)

    def test_abandoned_iterator_replays_epoch_with_a_clean_frontier(self) -> None:
        loader = DataLoader(PublicDataset(), batch_size=1, num_workers=2, seed=41)
        first_iterator = iter(loader)
        first_sample = next(first_iterator)
        try:
            replayed = list(loader)
            with self.assertRaisesRegex(RuntimeError, "no longer active"):
                next(first_iterator)
        finally:
            loader.close()

        self.assertEqual(int(first_sample["index"].item()), 0)
        self.assertEqual([int(batch["index"].item()) for batch in replayed], [0, 1, 2, 3])
        self.assertEqual(first_sample["torch"], replayed[0]["torch"])

    def test_construction_prepares_workers_and_retains_shuffled_probe(self) -> None:
        loader = DataLoader(list(range(8)), batch_size=2, shuffle=True, num_workers=2, seed=43)
        try:
            self.assertEqual(len(loader._process_pool.worker_pids), 2)
            first = [int(value) for batch in loader for value in batch]
            second = [int(value) for batch in loader for value in batch]
        finally:
            loader.close()

        expected_first = [
            _hyperloader._permutation_index(43, 0, 8, position)
            for position in range(8)
        ]
        expected_second = [
            _hyperloader._permutation_index(43, 1, 8, position)
            for position in range(8)
        ]
        self.assertEqual(first, expected_first)
        self.assertEqual(second, expected_second)

    def test_severed_process_dispatch_breaks_the_public_path(self) -> None:
        loader = DataLoader(PublicDataset(), batch_size=1, num_workers=1, seed=47)
        try:
            with mock.patch.object(
                ProcessPool, "try_submit", side_effect=RuntimeError("severed dispatch")
            ):
                with self.assertRaisesRegex(RuntimeError, "severed dispatch"):
                    iter(loader)
        finally:
            loader.close()

    def test_drop_last_excludes_tail_from_reusable_frontier(self) -> None:
        loader = DataLoader(list(range(5)), batch_size=2, drop_last=True, num_workers=2, seed=53)
        try:
            first = [[int(value) for value in batch] for batch in loader]
            second = [[int(value) for value in batch] for batch in loader]
        finally:
            loader.close()

        self.assertEqual(first, [[0, 1], [2, 3]])
        self.assertEqual(second, first)


if __name__ == "__main__":
    unittest.main()

"""Persistent process execution through the installed public package."""

from __future__ import annotations

import os
import random
import tempfile
import time
import unittest
from functools import partial
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.utils.data import get_worker_info

from hyperloader import DataLoader, _hyperloader
from hyperloader.process import ProcessPool
from hyperloader.process.random_surface import PhiloxRandom


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


class NumpyDataset:
    """Return contiguous NumPy rows through the black-box public path."""

    def __init__(self, length: int = 8) -> None:
        self.values = np.arange(length * 4, dtype=np.int64).reshape(length, 4)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.values[index]


class TensorDataset:
    """Return views from one parent-owned tensor storage."""

    def __init__(self, length: int = 8) -> None:
        self.values = torch.arange(length * 4, dtype=torch.int64).reshape(length, 4)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.values[index]


class NumpyRandomDataset:
    """Expose per-sample seeded NumPy draws in a homogeneous array."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> np.ndarray:
        return np.array([index, np.random.randint(0, 2**31)], dtype=np.int64)


class NumpyFailureDataset(NumpyDataset):
    """Raise between homogeneous rows to verify exact failure placement."""

    def __getitem__(self, index: int) -> np.ndarray:
        if index == 1:
            raise ValueError("array position one failed")
        return super().__getitem__(index)


class BootBindingDataset:
    """Record whether dataset unpickling observes rebound RNG callables."""

    def __init__(self) -> None:
        self.marker = True

    def __len__(self) -> int:
        return 1

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self.random_bound = isinstance(random.random.__self__, PhiloxRandom)
        self.numpy_bound = isinstance(np.random.random.__self__, np.random.Generator)

    def __getitem__(self, _index: int) -> tuple[bool, bool]:
        return self.random_bound, self.numpy_bound


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


def record_worker_init(directory: str, _worker_id: int) -> None:
    """Record the once-per-process initialization view."""
    info = get_worker_info()
    (Path(directory) / f"worker-{info.id}.log").write_text(
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


def wait_for_signal(pool: ProcessPool, timeout: float = 1.0) -> bool:
    """Wait across liveness intervals until one event or the caller deadline."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pool.wait_for_completion(deadline):
            return True
    return False


class ProcessPoolTest(unittest.TestCase):
    """Exercise persistence, RNG installation, errors, and public delivery."""

    def test_per_sample_rng_reproduces_across_fresh_pools(self) -> None:
        first = ProcessPool(RandomDataset(), 2, 17, 0, 0, 0)
        second = ProcessPool(RandomDataset(), 2, 17, 0, 0, 0)
        try:
            first_values = [
                first.execute(0, position, position) for position in range(4)
            ]
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
            expected_seed = _hyperloader._sample_rng_context(17, 0, position)[0]
            self.assertEqual(value["seed"], expected_seed)

    def test_dataset_unpickling_observes_rebound_rng_callables(self) -> None:
        pool = ProcessPool(BootBindingDataset(), 1, 31, 0, 0, 0)
        try:
            self.assertEqual(pool.execute(0, 0, 0), (True, True))
        finally:
            pool.close()

    def test_public_loader_batches_numpy_rows_and_partial_tail(self) -> None:
        dataset = NumpyDataset(5)
        loader = DataLoader(dataset, batch_size=2, num_workers=2, seed=19)
        try:
            batches = list(loader)
        finally:
            loader.close()

        expected = [
            torch.utils.data.default_collate([dataset[index] for index in range(0, 2)]),
            torch.utils.data.default_collate([dataset[index] for index in range(2, 4)]),
            torch.utils.data.default_collate([dataset[4]]),
        ]
        self.assertEqual(len(batches), len(expected))
        self.assertTrue(
            all(torch.equal(left, right) for left, right in zip(batches, expected))
        )

    def test_numpy_rows_use_one_native_command_per_batch(self) -> None:
        loader = DataLoader(NumpyDataset(5), batch_size=2, num_workers=2, seed=23)
        try:
            self.assertEqual(loader._process_pool.batch_size, 2)
            with mock.patch.object(
                loader._process_pool,
                "try_submit",
                wraps=loader._process_pool.try_submit,
            ) as submit:
                batches = list(loader)
        finally:
            loader.close()

        self.assertEqual(len(batches), 3)
        self.assertEqual(submit.call_count, 3)
        self.assertEqual([call.args[1] for call in submit.call_args_list], [0, 1, 2])
        self.assertEqual(
            [call.kwargs["batch_len"] for call in submit.call_args_list],
            [2, 2, 1],
        )

    def test_scalar_batch_boundary_wakes_the_owner_control_pipe(self) -> None:
        pool = ProcessPool(PublicDataset(), 1, 23, 0, 0, 0, batch_size=2)
        try:
            self.assertTrue(pool.try_submit(0, 0, 0, 0))
            self.assertTrue(pool.try_submit(0, 1, 1, 0))
            self.assertTrue(wait_for_signal(pool))
            completions = []
            while len(completions) < 2:
                completion = pool.try_receive(0)
                if completion is not None:
                    completions.append(completion)
        finally:
            pool.close()

        self.assertEqual([item[0] for item in completions], [0, 1])

    def test_native_completion_wakes_the_owner_control_pipe(self) -> None:
        pool = ProcessPool(NumpyDataset(2), 1, 23, 0, 0, 0, batch_size=2)
        try:
            self.assertTrue(pool.try_submit(0, 0, 0, 0, batch_len=2))
            self.assertTrue(wait_for_signal(pool))
            completion = pool.try_receive(0)
            self.assertIsNotNone(completion)
            position, status, payload = completion
            value = pool.decode_batch(status, payload, 0)
        finally:
            pool.close()

        self.assertEqual(position, 0)
        self.assertEqual(status, 2)
        self.assertTrue(torch.equal(value, torch.arange(8).reshape(2, 4)))

    def test_non_array_probe_retains_scalar_storage_transport(self) -> None:
        loader = DataLoader(PublicDataset(), batch_size=2, num_workers=2, seed=29)
        try:
            self.assertIsNone(loader._process_pool.batch_size)
            batches = list(loader)
        finally:
            loader.close()

        self.assertEqual(sum(batch["index"].numel() for batch in batches), 4)

    def test_each_tensor_worker_receives_an_independent_storage_token(self) -> None:
        dataset = TensorDataset()
        loader = DataLoader(dataset, batch_size=2, num_workers=4, seed=29)
        try:
            batches = list(loader)
        finally:
            loader.close()

        self.assertTrue(torch.equal(torch.cat(batches), dataset.values))

    def test_numpy_batching_preserves_per_sample_rng(self) -> None:
        first = DataLoader(NumpyRandomDataset(), batch_size=2, num_workers=2, seed=29)
        second = DataLoader(NumpyRandomDataset(), batch_size=2, num_workers=2, seed=29)
        try:
            first_batches = list(first)
            second_batches = list(second)
        finally:
            first.close()
            second.close()

        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(first_batches, second_batches)
            )
        )

    def test_numpy_batching_retains_the_failing_sample_position(self) -> None:
        loader = DataLoader(NumpyFailureDataset(), batch_size=2, num_workers=2, seed=31)
        try:
            with self.assertRaisesRegex(ValueError, "array position one failed"):
                next(iter(loader))
        finally:
            loader.close()

    def test_worker_init_runs_once_with_no_sample_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pool = ProcessPool(
                RandomDataset(),
                2,
                23,
                0,
                0,
                0,
                worker_init_fn=partial(record_worker_init, directory),
            )
            try:
                for position in range(4):
                    pool.execute(0, position, position)
                expected_pids = set(pool.worker_pids)
            finally:
                pool.close()

            records = [
                path.read_text(encoding="utf-8").strip()
                for path in sorted(Path(directory).glob("worker-*.log"))
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual({int(row.split(":")[0]) for row in records}, expected_pids)
            self.assertTrue(all(row.endswith(":None") for row in records))

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

    def test_delivering_abandoned_iterator_advances_with_a_clean_frontier(self) -> None:
        loader = DataLoader(PublicDataset(), batch_size=1, num_workers=2, seed=41)
        first_iterator = iter(loader)
        first_sample = next(first_iterator)
        try:
            with self.assertWarnsRegex(UserWarning, "advanced the epoch"):
                resumed = list(loader)
            with self.assertRaisesRegex(RuntimeError, "no longer active"):
                next(first_iterator)
        finally:
            loader.close()

        self.assertEqual(int(first_sample["index"].item()), 0)
        self.assertEqual(
            [int(batch["index"].item()) for batch in resumed], [0, 1, 2, 3]
        )
        self.assertNotEqual(first_sample["torch"], resumed[0]["torch"])

    def test_empty_abandoned_iterator_replays_without_a_notice(self) -> None:
        loader = DataLoader(PublicDataset(), batch_size=1, num_workers=2, seed=41)
        first_iterator = iter(loader)
        try:
            with mock.patch("hyperloader.api.warnings.warn") as warn:
                replayed = list(loader)
            with self.assertRaisesRegex(RuntimeError, "no longer active"):
                next(first_iterator)
        finally:
            loader.close()

        warn.assert_not_called()
        expected_seed = _hyperloader._sample_rng_context(41, 0, 0)[0]
        generator = torch.Generator().manual_seed(expected_seed)
        expected_draw = torch.rand((), generator=generator).item()
        self.assertEqual(replayed[0]["torch"].item(), expected_draw)

    def test_set_epoch_explicitly_replays_partial_epoch(self) -> None:
        loader = DataLoader(PublicDataset(), batch_size=1, num_workers=2, seed=41)
        first_iterator = iter(loader)
        first_sample = next(first_iterator)
        loader.set_epoch(0)
        try:
            with mock.patch("warnings.warn") as warn:
                replayed = list(loader)
        finally:
            loader.close()

        warn.assert_not_called()
        self.assertEqual(first_sample["torch"], replayed[0]["torch"])

    def test_construction_prepares_workers_and_retains_shuffled_probe(self) -> None:
        loader = DataLoader(
            list(range(8)), batch_size=2, shuffle=True, num_workers=2, seed=43
        )
        try:
            self.assertEqual(len(loader._process_pool.worker_pids), 2)
            first = [int(value) for batch in loader for value in batch]
            second = [int(value) for batch in loader for value in batch]
        finally:
            loader.close()

        expected_first = [
            _hyperloader._permutation_index(43, 0, 8, position) for position in range(8)
        ]
        expected_second = [
            _hyperloader._permutation_index(43, 1, 8, position) for position in range(8)
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
        loader = DataLoader(
            list(range(5)), batch_size=2, drop_last=True, num_workers=2, seed=53
        )
        try:
            first = [[int(value) for value in batch] for batch in loader]
            second = [[int(value) for value in batch] for batch in loader]
        finally:
            loader.close()

        self.assertEqual(first, [[0, 1], [2, 3]])
        self.assertEqual(second, first)


if __name__ == "__main__":
    unittest.main()

"""Calling-process parity, state, and isolation tests."""

from __future__ import annotations

import random
import unittest

import numpy as np
import torch
from hyperloader import DataLoader
from hyperloader.verify import _bit_equal
from torch.utils.data import get_worker_info


class SeededDataset:
    """Draw from every seeded CPU global and sanctioned accessor."""

    def __len__(self) -> int:
        return 11

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "index": index,
            "numpy": np.asarray(
                [np.random.random(), np.random.randint(0, 1 << 20)],
                dtype=np.float64,
            ),
            "python": (random.random(), random.getrandbits(27)),
            "torch": torch.rand(3),
        }


class WorkerViewDataset:
    """Expose the calling-process worker identity contract."""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[int, bool]:
        return index, get_worker_info() is None


class WorkerViewIterable:
    """Expose worker identity while iterating in the calling process."""

    def __iter__(self):
        yield get_worker_info() is None
        yield get_worker_info() is None


class FailingDataset:
    """Raise after consuming every seeded global surface."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> int:
        random.random()
        np.random.random()
        torch.rand(())
        raise ValueError(f"failure at {index}")


class FixedSampler:
    """Yield one repeatable custom sample order."""

    def __iter__(self):
        return iter((8, 2, 5, 0, 7))

    def __len__(self) -> int:
        return 5


class FixedBatchSampler:
    """Yield variable repeatable batches."""

    def __iter__(self):
        return iter(((4, 1, 8), (0,), (3, 7)))

    def __len__(self) -> int:
        return 3


def _rng_state() -> tuple[object, object, torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.default_generator.get_state()


def seeded_collate(values: list[object]) -> dict[str, object]:
    """Expose the batch-level installed RNG stream."""
    return {
        "numpy": np.random.random(),
        "python": random.random(),
        "torch": torch.rand(2),
        "values": torch.utils.data.default_collate(values),
    }


def _assert_rng_state(
    test: unittest.TestCase,
    expected: tuple[object, object, torch.Tensor],
) -> None:
    actual = _rng_state()
    test.assertEqual(actual[0], expected[0])
    expected_numpy = expected[1]
    actual_numpy = actual[1]
    test.assertEqual(actual_numpy[0], expected_numpy[0])
    test.assertTrue(np.array_equal(actual_numpy[1], expected_numpy[1]))
    test.assertEqual(actual_numpy[2:], expected_numpy[2:])
    test.assertTrue(torch.equal(actual[2], expected[2]))


class InProcessIteratorTest(unittest.TestCase):
    """Preserve outputs and trainer state in the calling-process tier."""

    def test_seeded_stream_matches_process_for_multiple_seeds(self) -> None:
        for seed in (0, 61, 2**63 + 9):
            with self.subTest(seed=seed):
                process = DataLoader(
                    SeededDataset(), batch_size=3, num_workers=2, seed=seed
                )
                local = DataLoader(
                    SeededDataset(), batch_size=3, num_workers=0, seed=seed
                )
                try:
                    self.assertTrue(_bit_equal(list(process), list(local)))
                finally:
                    process.close()
                    local.close()

    def test_each_batch_restores_all_cpu_globals(self) -> None:
        loader = DataLoader(SeededDataset(), batch_size=4, num_workers=0, seed=73)
        iterator = iter(loader)
        try:
            expected = _rng_state()
            next(iterator)
            _assert_rng_state(self, expected)
            next(iterator)
            _assert_rng_state(self, expected)
        finally:
            loader.close()

    def test_user_collation_matches_process_and_restores_globals(self) -> None:
        process = DataLoader(
            range(9),
            batch_size=3,
            num_workers=2,
            seed=75,
            collate_fn=seeded_collate,
        )
        local = DataLoader(
            range(9),
            batch_size=3,
            num_workers=0,
            seed=75,
            collate_fn=seeded_collate,
        )
        expected = _rng_state()
        try:
            self.assertTrue(_bit_equal(list(process), list(local)))
            _assert_rng_state(self, expected)
        finally:
            process.close()
            local.close()

    def test_exception_restores_globals_and_keeps_original_type(self) -> None:
        loader = DataLoader(FailingDataset(), num_workers=0, seed=79)
        expected = _rng_state()
        with self.assertRaisesRegex(ValueError, "failure at 0"):
            next(iter(loader))
        _assert_rng_state(self, expected)

    def test_worker_info_is_none_and_initializer_does_not_run(self) -> None:
        calls = []
        map_loader = DataLoader(
            WorkerViewDataset(),
            batch_size=2,
            num_workers=0,
            worker_init_fn=calls.append,
        )
        iterable_loader = DataLoader(
            WorkerViewIterable(),
            batch_size=2,
            num_workers=0,
            worker_init_fn=calls.append,
        )
        try:
            map_batch = next(iter(map_loader))
            iterable_batch = next(iter(iterable_loader))
            self.assertEqual(map_batch[1].tolist(), [True, True])
            self.assertEqual(iterable_batch.tolist(), [True, True])
            self.assertEqual(calls, [])
        finally:
            map_loader.close()
            iterable_loader.close()

    def test_shuffle_resume_and_user_sampler_grouping(self) -> None:
        full = DataLoader(range(11), batch_size=3, shuffle=True, num_workers=0, seed=83)
        source = DataLoader(
            range(11), batch_size=3, shuffle=True, num_workers=0, seed=83
        )
        source_iterator = iter(source)
        prefix = [next(source_iterator), next(source_iterator)]
        state = source.state_dict()
        resumed = DataLoader(
            range(11), batch_size=3, shuffle=True, num_workers=0, seed=97
        )
        resumed.load_state_dict(state)
        sampler = DataLoader(
            range(10),
            batch_size=2,
            sampler=FixedSampler(),
            num_workers=0,
            seed=89,
        )
        batch_sampler = DataLoader(
            range(10),
            batch_sampler=FixedBatchSampler(),
            num_workers=0,
            seed=101,
        )
        try:
            self.assertTrue(_bit_equal(list(full), prefix + list(resumed)))
            self.assertEqual(
                [batch.tolist() for batch in sampler], [[8, 2], [5, 0], [7]]
            )
            self.assertEqual(
                [batch.tolist() for batch in batch_sampler],
                [[4, 1, 8], [0], [3, 7]],
            )
        finally:
            full.close()
            source.close()
            resumed.close()
            sampler.close()
            batch_sampler.close()


if __name__ == "__main__":
    unittest.main()

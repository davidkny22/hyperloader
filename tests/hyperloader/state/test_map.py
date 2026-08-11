"""Map-style coordinate capture and exact continuation checks."""

from __future__ import annotations

import copy
import unittest
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from hyperloader import DataLoader


class RangeDataset:
    """Provide a picklable black-box map dataset for process and thread checks."""

    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor([index, index + 100], dtype=torch.int64)


class FixedSampler:
    """Yield one configured deterministic index stream."""

    def __init__(self, indices: list[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Any:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class FixedBatchSampler:
    """Yield configured variable batches without automatic regrouping."""

    def __init__(self, batches: list[list[int]]) -> None:
        self.batches = batches

    def __iter__(self) -> Any:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


class DriftingSampler:
    """Change order on the second iteration without changing public identity."""

    def __init__(self, indices: list[int]) -> None:
        self.indices = indices
        self._iterations = 0

    def __iter__(self) -> Any:
        self._iterations += 1
        values = self.indices if self._iterations == 1 else reversed(self.indices)
        return iter(values)

    def __len__(self) -> int:
        return len(self.indices)


def _assert_batches_equal(
    case: unittest.TestCase, actual: list[Any], expected: list[Any]
) -> None:
    case.assertEqual(len(actual), len(expected))
    for left, right in zip(actual, expected, strict=True):
        case.assertTrue(torch.equal(left, right))


class MapCoordinateStateTest(unittest.TestCase):
    """Exercise native delivered-prefix state through the public loader."""

    def test_tensor_state_resumes_exact_remaining_batches(self) -> None:
        dataset = torch.arange(40, dtype=torch.int64).reshape(20, 2)
        loader = DataLoader(dataset, batch_size=4, shuffle=True, seed=17, num_workers=1)
        iterator = iter(loader)
        delivered = [next(iterator), next(iterator)]
        state = loader.state_dict()
        expected = list(iterator)

        resumed = DataLoader(
            dataset, batch_size=4, shuffle=True, seed=999, num_workers=1
        )
        resumed.load_state_dict(state)
        actual = list(resumed)

        self.assertEqual(state["root_seed"], 17)
        self.assertEqual(state["epoch"], 0)
        self.assertEqual(state["cursor"], len(delivered))
        self.assertEqual(state["B_g"], 4)
        self.assertEqual(state["sampler_checksum"], 0)
        self.assertEqual(resumed.root_seed, 17)
        _assert_batches_equal(self, actual, expected)

    def test_process_state_discards_speculation_and_resumes_prefix(self) -> None:
        dataset = RangeDataset(18)
        loader = DataLoader(dataset, batch_size=3, seed=23, num_workers=2)
        iterator = iter(loader)
        next(iterator)
        next(iterator)
        state = loader.state_dict()
        expected = list(iterator)

        resumed = DataLoader(dataset, batch_size=3, seed=99, num_workers=2)
        resumed.load_state_dict(state)
        actual = list(resumed)

        self.assertEqual(state["cursor"], 2)
        _assert_batches_equal(self, actual, expected)
        loader.close()
        resumed.close()

    def test_thread_state_resumes_from_the_same_coordinate(self) -> None:
        dataset = RangeDataset(15)
        loader = DataLoader(
            dataset,
            batch_size=3,
            seed=29,
            num_workers=2,
            thread_safe=True,
        )
        iterator = iter(loader)
        next(iterator)
        state = loader.state_dict()
        expected = list(iterator)

        resumed = DataLoader(
            dataset,
            batch_size=3,
            seed=31,
            num_workers=2,
            thread_safe=True,
        )
        resumed.load_state_dict(state)

        _assert_batches_equal(self, list(resumed), expected)
        loader.close()
        resumed.close()

    def test_structured_state_resumes_batch_native_delivery(self) -> None:
        dataset = np.arange(24, dtype=np.int64).reshape(12, 2)
        loader = DataLoader(dataset, batch_size=3, seed=33, num_workers=2)
        iterator = iter(loader)
        next(iterator)
        state = loader.state_dict()
        expected = list(iterator)

        resumed = DataLoader(dataset, batch_size=3, seed=35, num_workers=2)
        resumed.load_state_dict(state)

        self.assertIsNone(resumed._process_pool)
        _assert_batches_equal(self, list(resumed), expected)
        loader.close()
        resumed.close()

    def test_completed_iterator_captures_the_next_epoch_origin(self) -> None:
        loader = DataLoader(
            torch.arange(8), batch_size=2, shuffle=True, seed=37, num_workers=1
        )
        list(loader)

        state = loader.state_dict()

        self.assertEqual(state["epoch"], 1)
        self.assertEqual(state["cursor"], 0)

    def test_set_epoch_overrides_a_loaded_cursor_for_explicit_replay(self) -> None:
        dataset = torch.arange(12)
        source = DataLoader(dataset, batch_size=3, shuffle=True, seed=41, num_workers=1)
        iterator = iter(source)
        next(iterator)
        state = source.state_dict()

        resumed = DataLoader(
            dataset, batch_size=3, shuffle=True, seed=41, num_workers=1
        )
        resumed.load_state_dict(state)
        resumed.set_epoch(0)
        replay = list(resumed)

        fresh = DataLoader(dataset, batch_size=3, shuffle=True, seed=41, num_workers=1)
        _assert_batches_equal(self, replay, list(fresh))

    def test_fingerprint_mismatch_names_the_changed_global_batch(self) -> None:
        dataset = torch.arange(12)
        source = DataLoader(dataset, batch_size=2, seed=43, num_workers=1)
        state = source.state_dict()
        changed = DataLoader(dataset, batch_size=3, seed=43, num_workers=1)

        with self.assertRaisesRegex(ValueError, "placement.B_g"):
            changed.load_state_dict(state)

    def test_state_rejects_tampered_fingerprint_and_native_checksum(self) -> None:
        loader = DataLoader(torch.arange(8), batch_size=2, seed=47, num_workers=1)
        state = loader.state_dict()
        bad_digest = copy.deepcopy(state)
        bad_digest["fingerprint"]["digest"] = "0" * 64
        bad_checksum = copy.deepcopy(state)
        bad_checksum["sampler_checksum"] = 1

        with self.assertRaisesRegex(ValueError, "digest"):
            loader.load_state_dict(bad_digest)
        with self.assertRaisesRegex(ValueError, "sampler_checksum=0"):
            loader.load_state_dict(bad_checksum)

    def test_cursor_beyond_epoch_is_rejected_on_iteration(self) -> None:
        loader = DataLoader(torch.arange(8), batch_size=2, seed=53, num_workers=1)
        state = loader.state_dict()
        state["cursor"] = 5
        loader.load_state_dict(state)

        with self.assertRaisesRegex(ValueError, "exceeds 4"):
            iter(loader)

    def test_user_sampler_resumes_exactly_under_checksum(self) -> None:
        dataset = RangeDataset(10)
        sampler = FixedSampler([7, 1, 9, 0, 4, 2, 8, 5])
        loader = DataLoader(
            dataset, batch_size=2, sampler=sampler, seed=59, num_workers=2
        )
        iterator = iter(loader)
        next(iterator)
        next(iterator)
        state = loader.state_dict()
        expected = list(iterator)

        resumed = DataLoader(
            dataset,
            batch_size=2,
            sampler=FixedSampler(list(sampler.indices)),
            seed=61,
            num_workers=2,
        )
        resumed.load_state_dict(state)

        self.assertEqual(state["B_g"], 0)
        self.assertNotEqual(state["sampler_checksum"], 0)
        _assert_batches_equal(self, list(resumed), expected)
        loader.close()
        resumed.close()

    def test_batch_sampler_preserves_variable_batches_across_resume(self) -> None:
        dataset = RangeDataset(8)
        batches = [[6, 1, 3], [0], [7, 2], [5, 4]]
        loader = DataLoader(
            dataset,
            batch_sampler=FixedBatchSampler(batches),
            seed=67,
            num_workers=2,
        )
        iterator = iter(loader)
        next(iterator)
        state = loader.state_dict()
        expected = list(iterator)

        resumed = DataLoader(
            dataset,
            batch_sampler=FixedBatchSampler(copy.deepcopy(batches)),
            seed=71,
            num_workers=2,
        )
        resumed.load_state_dict(state)
        actual = list(resumed)

        self.assertEqual([len(batch) for batch in actual], [1, 2, 2])
        _assert_batches_equal(self, actual, expected)
        loader.close()
        resumed.close()

    def test_nondeterministic_sampler_names_both_checksum_causes(self) -> None:
        loader = DataLoader(
            RangeDataset(8),
            batch_size=2,
            sampler=DriftingSampler(list(range(8))),
            seed=73,
            num_workers=2,
        )
        iterator = iter(loader)
        next(iterator)
        state = loader.state_dict()
        loader.load_state_dict(state)

        with self.assertRaisesRegex(ValueError, "nondeterministic.*different rank"):
            iter(loader)
        loader.close()

    def test_user_sampler_and_batch_sampler_match_torch_grouping(self) -> None:
        dataset = RangeDataset(9)
        indices = [8, 2, 5, 0, 7, 1, 6]
        batches = [[4, 1, 8], [0], [3, 7]]

        hyper_sampler = DataLoader(
            dataset,
            batch_size=3,
            sampler=FixedSampler(indices),
            seed=79,
            num_workers=2,
        )
        torch_sampler = TorchDataLoader(
            dataset, batch_size=3, sampler=FixedSampler(indices), num_workers=0
        )
        hyper_batches = DataLoader(
            dataset,
            batch_sampler=FixedBatchSampler(batches),
            seed=83,
            num_workers=2,
        )
        torch_batches = TorchDataLoader(
            dataset,
            batch_sampler=FixedBatchSampler(batches),
            num_workers=0,
        )

        _assert_batches_equal(self, list(hyper_sampler), list(torch_sampler))
        _assert_batches_equal(self, list(hyper_batches), list(torch_batches))
        hyper_sampler.close()
        hyper_batches.close()


if __name__ == "__main__":
    unittest.main()

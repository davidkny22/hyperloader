"""Map-style coordinate capture and exact continuation checks."""

from __future__ import annotations

import copy
import unittest
from typing import Any

import numpy as np
import torch

from hyperloader import DataLoader


class RangeDataset:
    """Provide a picklable black-box map dataset for process and thread checks."""

    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor([index, index + 100], dtype=torch.int64)


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


if __name__ == "__main__":
    unittest.main()

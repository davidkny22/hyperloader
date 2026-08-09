"""Public contiguous-tensor iterator contract checks."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from hyperloader import DataLoader, _hyperloader
from hyperloader.tensor import TensorIterator


class TensorIteratorTest(unittest.TestCase):
    """Exercise exact delivery through the installed public entry point."""

    def test_contiguous_batch_is_a_storage_view(self) -> None:
        dataset = torch.arange(30, dtype=torch.int64).reshape(10, 3)
        loader = DataLoader(dataset, batch_size=4, num_workers=2)

        batch = next(iter(loader))

        self.assertTrue(torch.equal(batch, torch.stack([dataset[i] for i in range(4)])))
        self.assertEqual(batch.stride(), (3, 1))
        self.assertEqual(
            batch.untyped_storage().data_ptr(),
            dataset.untyped_storage().data_ptr(),
        )
        self.assertIsNone(loader._process_pool)

    def test_shuffled_batch_matches_native_sampler_order(self) -> None:
        dataset = torch.arange(24, dtype=torch.int64).reshape(8, 3)
        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, seed=11)
        indices = [
            _hyperloader._permutation_index(11, 0, 8, position) for position in range(4)
        ]

        self.assertTrue(torch.equal(next(iter(loader)), dataset[indices]))

    def test_drop_last_clips_before_delivery_and_advances_epoch(self) -> None:
        loader = DataLoader(
            torch.arange(15).reshape(5, 3), batch_size=2, drop_last=True, num_workers=1
        )
        iterator = iter(loader)

        self.assertEqual([batch.shape[0] for batch in iterator], [2, 2])
        self.assertTrue(iterator.complete)
        self.assertEqual(loader._epoch, 1)

    def test_public_route_has_wiring_teeth(self) -> None:
        loader = DataLoader(torch.arange(6).reshape(2, 3), num_workers=1)
        with mock.patch.object(
            TensorIterator,
            "__next__",
            side_effect=RuntimeError("severed tensor route"),
        ):
            with self.assertRaisesRegex(RuntimeError, "severed tensor route"):
                next(iter(loader))


if __name__ == "__main__":
    unittest.main()

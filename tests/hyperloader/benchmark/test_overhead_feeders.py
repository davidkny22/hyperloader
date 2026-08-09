"""Public and counterfactual feeder equivalence checks."""

from __future__ import annotations

import unittest

import torch

from benches.overhead_feeders import (
    LoaderFeeder,
    ResidentFeeder,
    fixed_text_tensor,
)


class OverheadFeederTest(unittest.TestCase):
    """Verify both paired arms deliver identical resident tensors."""

    def test_public_loader_and_counterfactual_match_without_process_pool(self) -> None:
        dataset = fixed_text_tensor(2)
        loader = LoaderFeeder(dataset)
        resident = ResidentFeeder(dataset)
        try:
            for _ in range(2):
                self.assertTrue(torch.equal(loader.next_batch(), resident.next_batch()))
            self.assertIsNone(loader._loader._process_pool)
        finally:
            loader.close()


if __name__ == "__main__":
    unittest.main()

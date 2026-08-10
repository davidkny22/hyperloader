"""Offline thread-safety evidence tests."""

from __future__ import annotations

import random
import unittest

import numpy as np
import torch
from hyperloader import rng, verify


class DeclaredDataset:
    """Produce nested values from sanctioned per-sample generators."""

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "index": index,
            "numpy": np.asarray([rng("numpy").random()], dtype=np.float64),
            "python": rng("random").random(),
            "torch": torch.rand(2, generator=rng()),
        }


class GlobalDrawDataset:
    """Violate the declaration by drawing from ambient Python state."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> tuple[int, float]:
        return index, random.random()


class VerifyTest(unittest.TestCase):
    """Exercise positive parity and the global-draw negative control."""

    def test_declared_dataset_passes_bit_exactly(self) -> None:
        report = verify(DeclaredDataset(), samples=8, seed=61, num_workers=2)

        self.assertEqual(
            report,
            {"bit_exact": True, "compared_samples": 8, "first_mismatch": None},
        )

    def test_global_rng_draw_fails_the_declaration_evidence(self) -> None:
        state = random.getstate()
        try:
            report = verify(GlobalDrawDataset(), samples=4, seed=67, num_workers=2)
        finally:
            random.setstate(state)

        self.assertFalse(report["bit_exact"])
        self.assertIsInstance(report["first_mismatch"], int)

    def test_verification_arguments_are_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "samples must be nonnegative"):
            verify(DeclaredDataset(), samples=-1)
        with self.assertRaisesRegex(ValueError, "num_workers must be positive"):
            verify(DeclaredDataset(), num_workers=0)


if __name__ == "__main__":
    unittest.main()

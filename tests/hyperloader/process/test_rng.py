"""Worker-local per-sample RNG installation checks."""

from __future__ import annotations

import random
import unittest

import numpy as np
import torch
from torch.utils.data import get_worker_info

from hyperloader import _hyperloader
from hyperloader.process.rng import WorkerRngContext


class WorkerRngContextTest(unittest.TestCase):
    """Exercise exact seeded globals and retained worker identity."""

    def setUp(self) -> None:
        self._random_state = random.getstate()
        self._numpy_state = np.random.get_state()
        self._torch_state = torch.get_rng_state()

    def tearDown(self) -> None:
        random.setstate(self._random_state)
        np.random.set_state(self._numpy_state)
        torch.set_rng_state(self._torch_state)

    def test_install_matches_each_derived_seed_stream(self) -> None:
        dataset = [0]
        context = WorkerRngContext(2, 4, dataset)
        try:
            torch_seed, random_seed, numpy_words = _hyperloader._sample_seed_words(
                17, 3, 11
            )
            context.install(17, 3, 11)
            info = get_worker_info()

            expected_random = random.Random(random_seed).random()
            expected_numpy = np.random.RandomState(np.asarray(numpy_words)).random()
            expected_torch = torch.rand(
                (), generator=torch.Generator().manual_seed(torch_seed)
            )

            self.assertEqual(info.id, 2)
            self.assertEqual(info.num_workers, 4)
            self.assertEqual(info.seed, torch_seed)
            self.assertIs(info.dataset, dataset)
            self.assertEqual(random.random(), expected_random)
            self.assertEqual(np.random.random(), expected_numpy)
            self.assertEqual(torch.rand(()).item(), expected_torch.item())
        finally:
            context.clear()

        self.assertIsNone(get_worker_info())

    def test_install_preserves_each_worker_info_value(self) -> None:
        context = WorkerRngContext(0, 1, [0])
        try:
            context.install(5, 0, 0)
            first = get_worker_info()
            context.install(5, 0, 1)
            second = get_worker_info()

            self.assertIsNot(first, second)
            self.assertNotEqual(first.seed, second.seed)
        finally:
            context.clear()


if __name__ == "__main__":
    unittest.main()

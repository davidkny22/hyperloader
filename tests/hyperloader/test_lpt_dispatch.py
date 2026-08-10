"""Public-path checks for profile-driven early dispatch."""

from __future__ import annotations

import time
import unittest
from unittest import mock

import numpy as np

from hyperloader import DataLoader
from hyperloader.process.pool import ProcessPool


class SkewedCostDataset:
    """Make one stable position dominate the measured execution costs."""

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> int:
        time.sleep(0.03 if index == 3 else 0.0001)
        return index


class SkewedArrayDataset:
    """Make the second array batch dominate its command-level estimate."""

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> np.ndarray:
        time.sleep(0.03 if index == 2 else 0.0001)
        return np.asarray([index], dtype=np.int64)


class LptDispatchTest(unittest.TestCase):
    """Prove profiled execution reorders while delivered values do not."""

    def test_warm_profile_dispatches_slow_position_first(self) -> None:
        loader = DataLoader(
            SkewedCostDataset(),
            batch_size=1,
            num_workers=2,
            seed=17,
        )
        submitted: list[int] = []
        original_submit = ProcessPool.try_submit

        def recording_submit(pool: ProcessPool, *args: object, **kwargs: object) -> bool:
            accepted = original_submit(pool, *args, **kwargs)
            if accepted:
                submitted.append(int(args[1]))
            return accepted

        try:
            self.assertEqual([int(value.item()) for value in loader], list(range(8)))
            with mock.patch.object(ProcessPool, "try_submit", recording_submit):
                delivered = [int(value.item()) for value in loader]
        finally:
            loader.close()

        self.assertEqual(delivered, list(range(8)))
        self.assertEqual(submitted[0], 3)

    def test_batch_transport_ranks_aggregate_sample_cost(self) -> None:
        loader = DataLoader(
            SkewedArrayDataset(),
            batch_size=2,
            num_workers=2,
            seed=17,
        )
        submitted: list[int] = []
        original_submit = ProcessPool.try_submit

        def recording_submit(pool: ProcessPool, *args: object, **kwargs: object) -> bool:
            accepted = original_submit(pool, *args, **kwargs)
            if accepted:
                submitted.append(int(args[1]))
            return accepted

        try:
            first = [int(item) for batch in loader for item in batch.flatten().tolist()]
            self.assertEqual(loader._process_pool.batch_size, 2)
            estimates = [loader._cost_profile.estimate(position) for position in range(4)]
            self.assertTrue(all(estimate is not None for estimate in estimates))
            self.assertGreater(
                sum(estimate or 0.0 for estimate in estimates[2:4]),
                sum(estimate or 0.0 for estimate in estimates[0:2]),
            )
            with mock.patch.object(ProcessPool, "try_submit", recording_submit):
                second = [
                    int(item) for batch in loader for item in batch.flatten().tolist()
                ]
        finally:
            loader.close()

        self.assertEqual(first, list(range(8)))
        self.assertEqual(second, list(range(8)))
        self.assertEqual(submitted[0], 1)


if __name__ == "__main__":
    unittest.main()

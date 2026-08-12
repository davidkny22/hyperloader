"""Installed public gate for bounded adaptive frontier behavior."""

from __future__ import annotations

import unittest

from hyperloader import DataLoader, HyperConfig
from hyperloader.config import SchedulerConfig


class FrontierBudgetGate(unittest.TestCase):
    """Prove formula bounds, adapted stalls, and named binding causes."""

    def test_budget_below_liveness_floor_keeps_two_batches(self) -> None:
        config = HyperConfig(scheduler=SchedulerConfig(frontier_budget=1))
        loader = DataLoader(range(8), batch_size=2, num_workers=2, config=config)
        try:
            self.assertEqual(loader._process_pool.frontier_ceiling, 4)
            self.assertTrue(loader._process_pool.frontier_budget_bound)
            self.assertEqual(sum(batch.numel() for batch in loader), 8)
        finally:
            loader.close()

    def test_profile_cache_off_retains_the_cold_rule(self) -> None:
        config = HyperConfig(scheduler=SchedulerConfig(profile_cache="off"))
        loader = DataLoader(range(8), batch_size=2, num_workers=2, config=config)
        try:
            self.assertIsNone(loader._cost_profile)
            self.assertEqual(
                [value.tolist() for value in loader], [[0, 1], [2, 3], [4, 5], [6, 7]]
            )
            self.assertEqual(loader._last_frontier_report["binding"], "cold-variance")
        finally:
            loader.close()


if __name__ == "__main__":
    unittest.main()

"""Fixed frontier and transport sizing tests."""

import unittest
from types import SimpleNamespace
from unittest import mock

from hyperloader import AUTO
from hyperloader.process.sizing import (
    delivery_length,
    frontier_budget,
    frontier_ceiling,
    frontier_depth,
    queue_capacity,
)


class ProcessSizingTest(unittest.TestCase):
    """Exercise the liveness floor and per-worker queue coverage."""

    def test_frontier_respects_two_batch_floor(self) -> None:
        loader = SimpleNamespace(
            batch_size=3,
            num_workers=2,
            prefetch_factor=AUTO,
            _cost_profile=None,
            _process_pool=None,
            config=SimpleNamespace(
                scheduler=SimpleNamespace(frontier_depth=AUTO, frontier_budget=AUTO),
                factors=SimpleNamespace(d_min=AUTO, f_var=8.0, f_safety=1.5),
            ),
        )

        self.assertEqual(frontier_depth(loader), 6)
        loader.config.scheduler.frontier_depth = 2
        self.assertEqual(frontier_depth(loader), 6)

    def test_profile_ratio_sets_active_depth_and_plan_ceiling(self) -> None:
        profile = SimpleNamespace(statistics=lambda: (10.0, 40.0, 8))
        loader = SimpleNamespace(
            batch_size=2,
            num_workers=3,
            prefetch_factor=AUTO,
            _cost_profile=profile,
            _process_pool=None,
            config=SimpleNamespace(
                scheduler=SimpleNamespace(frontier_depth=AUTO, frontier_budget=AUTO),
                factors=SimpleNamespace(d_min=AUTO, f_var=8.0, f_safety=1.5),
            ),
        )

        self.assertEqual(frontier_depth(loader), 18)
        self.assertEqual(frontier_ceiling(loader), 18)

    def test_explicit_budget_clips_only_above_the_liveness_floor(self) -> None:
        loader = SimpleNamespace(
            batch_size=4,
            num_workers=4,
            prefetch_factor=AUTO,
            _cost_profile=SimpleNamespace(statistics=lambda: (1.0, 20.0, 8)),
            _process_pool=SimpleNamespace(bytes_sample=100, frontier_ceiling=10),
            config=SimpleNamespace(
                scheduler=SimpleNamespace(frontier_depth=AUTO, frontier_budget=1_000),
                factors=SimpleNamespace(d_min=AUTO, f_var=8.0, f_safety=1.5),
            ),
        )

        self.assertEqual(frontier_depth(loader), 10)
        loader.config.scheduler.frontier_budget = 200
        self.assertEqual(frontier_depth(loader), 8)

    def test_prefetch_hint_seeds_depth_inside_the_plan_ceiling(self) -> None:
        loader = SimpleNamespace(
            batch_size=2,
            num_workers=3,
            prefetch_factor=3,
            _cost_profile=None,
            _process_pool=None,
            config=SimpleNamespace(
                scheduler=SimpleNamespace(frontier_depth=AUTO, frontier_budget=AUTO),
                factors=SimpleNamespace(d_min=AUTO, f_var=8.0, f_safety=1.5),
            ),
        )

        self.assertEqual(frontier_depth(loader), 18)

    def test_auto_frontier_budget_uses_the_named_memory_fraction(self) -> None:
        loader = SimpleNamespace(
            config=SimpleNamespace(
                scheduler=SimpleNamespace(frontier_budget=AUTO),
                factors=SimpleNamespace(f_mem=0.15),
            )
        )

        with mock.patch(
            "hyperloader.process.sizing.free_host_memory", return_value=10_000
        ):
            self.assertEqual(frontier_budget(loader), 1_500)

    def test_power_of_two_queues_cover_frontier(self) -> None:
        capacity = queue_capacity(17, 3)

        self.assertEqual(capacity & (capacity - 1), 0)
        self.assertGreaterEqual(capacity * 3, 17)

    def test_drop_last_clips_the_scheduled_range(self) -> None:
        loader = SimpleNamespace(
            _plan=SimpleNamespace(length=5), batch_size=2, drop_last=True
        )

        self.assertEqual(delivery_length(loader), 4)


if __name__ == "__main__":
    unittest.main()

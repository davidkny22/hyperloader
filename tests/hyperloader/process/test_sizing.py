"""Fixed frontier and transport sizing tests."""

import unittest
from types import SimpleNamespace

from hyperloader import AUTO
from hyperloader.process.sizing import delivery_length, frontier_depth, queue_capacity


class ProcessSizingTest(unittest.TestCase):
    """Exercise the liveness floor and per-worker queue coverage."""

    def test_frontier_respects_two_batch_floor(self) -> None:
        loader = SimpleNamespace(
            batch_size=3,
            config=SimpleNamespace(scheduler=SimpleNamespace(frontier_depth=AUTO)),
        )

        self.assertEqual(frontier_depth(loader), 6)
        loader.config.scheduler.frontier_depth = 2
        self.assertEqual(frontier_depth(loader), 6)

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

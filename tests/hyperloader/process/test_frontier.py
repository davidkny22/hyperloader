"""Adaptive frontier runtime checks."""

from __future__ import annotations

import unittest

from hyperloader.process.frontier import FrontierRuntime


class FrontierRuntimeTest(unittest.TestCase):
    """Exercise monotonic growth and bounded occupancy through the native scheduler."""

    def test_saturated_wait_grows_without_crossing_the_ceiling(self) -> None:
        frontier = FrontierRuntime(10, 2, 8, 2, 2, "cold-variance")
        for _ in range(2):
            position, worker = frontier.next_dispatch()
            frontier.mark_dispatched(position, worker)

        frontier.record_wait(100)
        for _ in range(2):
            position, worker = frontier.next_dispatch()
            frontier.mark_dispatched(position, worker)
        frontier.record_wait(200)
        report = frontier.report()

        self.assertEqual(report["initial_depth"], 2)
        self.assertEqual(report["final_depth"], 8)
        self.assertEqual(report["growth_events"], 2)
        self.assertLessEqual(report["max_occupied"], report["ceiling"])
        self.assertEqual(report["wait_ns"], 300)

    def test_unsaturated_wait_does_not_expand_the_frontier(self) -> None:
        frontier = FrontierRuntime(1, 2, 8, 1, 2, "cold-variance")
        position, worker = frontier.next_dispatch()
        frontier.mark_dispatched(position, worker)

        frontier.record_wait(100)

        self.assertEqual(frontier.report()["final_depth"], 2)

    def test_known_costs_dispatch_descending_with_position_ties(self) -> None:
        costs = {0: 10.0, 1: 100.0, 2: 100.0, 3: 20.0}
        frontier = FrontierRuntime(
            4,
            4,
            4,
            2,
            2,
            "profile-tail",
            costs.get,
        )

        order = []
        while (dispatch := frontier.next_dispatch()) is not None:
            position, worker = dispatch
            order.append(position)
            frontier.mark_dispatched(position, worker)

        self.assertEqual(order, [1, 2, 3, 0])

    def test_unknown_costs_retain_fifo_order(self) -> None:
        frontier = FrontierRuntime(3, 3, 3, 1, 2, "cold-variance", lambda _: None)

        order = []
        while (dispatch := frontier.next_dispatch()) is not None:
            position, worker = dispatch
            order.append(position)
            frontier.mark_dispatched(position, worker)

        self.assertEqual(order, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()

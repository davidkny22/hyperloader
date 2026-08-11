"""Delivered-batch bitmap state transitions."""

from __future__ import annotations

import unittest

from hyperloader.state import DeliveredBatchState


class DeliveredBatchStateTest(unittest.TestCase):
    """Keep completion-order state relative to its first delivery gap."""

    def test_gap_closure_advances_the_prefix_and_rebases_the_bitmap(self) -> None:
        state = DeliveredBatchState()
        state.mark(3)
        state.mark(1)
        self.assertEqual(state.base, 0)
        self.assertEqual(state.bitmap(), b"\x0a")

        state.mark(0)
        self.assertEqual(state.base, 2)
        self.assertEqual(state.bitmap(), b"\x02")

        state.mark(2)
        self.assertEqual(state.base, 4)
        self.assertEqual(state.bitmap(), b"")

    def test_duplicate_delivery_is_rejected(self) -> None:
        state = DeliveredBatchState(2)
        with self.assertRaisesRegex(RuntimeError, "repeated"):
            state.mark(1)
        state.mark(4)
        with self.assertRaisesRegex(RuntimeError, "repeated"):
            state.mark(4)


if __name__ == "__main__":
    unittest.main()

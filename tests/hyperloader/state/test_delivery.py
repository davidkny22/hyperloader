"""Delivered-batch bitmap state transitions."""

from __future__ import annotations

import unittest

from hyperloader.state import DeliveredBatchState, decode_delivered_bitmap


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

    def test_bitmap_decode_rejects_the_gap_and_out_of_range_bits(self) -> None:
        self.assertEqual(decode_delivered_bitmap(2, b"\x0a", 8), {3, 5})
        with self.assertRaisesRegex(ValueError, "bit zero"):
            decode_delivered_bitmap(2, b"\x01", 8)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            decode_delivered_bitmap(7, b"\x02", 8)


if __name__ == "__main__":
    unittest.main()

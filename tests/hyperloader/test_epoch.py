"""Map-style epoch transition checks."""

from __future__ import annotations

import unittest

from hyperloader.epoch import EpochState


class EpochStateTest(unittest.TestCase):
    """Exercise transitions that do not depend on an execution tier."""

    def test_restored_epoch_suppresses_abandonment_advance_once(self) -> None:
        state = EpochState()
        state.begin_iteration()
        state.mark_delivered(0)

        state.restore(7)

        self.assertFalse(state.begin_iteration())
        self.assertEqual(state.current, 7)

    def test_completed_iterator_advances_once(self) -> None:
        state = EpochState()
        state.begin_iteration()
        state.complete(0)
        state.complete(0)

        self.assertEqual(state.current, 1)

    def test_epoch_validation_rejects_ambiguous_values(self) -> None:
        state = EpochState()

        with self.assertRaisesRegex(TypeError, "integer"):
            state.set_epoch(True)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            state.set_epoch(-1)


if __name__ == "__main__":
    unittest.main()

"""Bounded source snapshot ring selection and pruning."""

from __future__ import annotations

import unittest

from hyperloader.iterable.snapshot import SnapshotRing


class SnapshotRingTest(unittest.TestCase):
    """Retain one delivered anchor plus future production states."""

    def test_discard_before_keeps_the_newest_delivered_anchor(self) -> None:
        ring = SnapshotRing(
            stateful=True,
            cadence=1,
            maximum_bytes=32,
            depth=4,
        )
        for arrival in (0, 2, 4, 6):
            self.assertTrue(ring.push(arrival, bytes([arrival])))

        ring.discard_before(5)

        self.assertIsNone(ring.select(3))
        self.assertEqual(ring.select(5).arrival, 4)
        self.assertEqual(ring.select(6).arrival, 6)


if __name__ == "__main__":
    unittest.main()

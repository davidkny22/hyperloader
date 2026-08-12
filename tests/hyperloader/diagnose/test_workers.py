"""Worker process resource observation."""

from __future__ import annotations

import os
import unittest

from hyperloader.diagnose.workers import snapshot_workers


class WorkerSnapshotTest(unittest.TestCase):
    """Read process counters without changing process state."""

    def test_current_process_has_cpu_and_resident_counters(self) -> None:
        observed = snapshot_workers((os.getpid(),))

        self.assertEqual(len(observed), 1)
        self.assertTrue(observed[0]["alive"])
        self.assertGreaterEqual(observed[0]["cpu_ns"], 0)
        self.assertGreater(observed[0]["rss_bytes"], 0)

    def test_missing_process_is_reported_without_raising(self) -> None:
        observed = snapshot_workers((2_147_483_647,))[0]

        self.assertFalse(observed["alive"])
        self.assertIsNone(observed["cpu_ns"])
        self.assertIsNone(observed["rss_bytes"])


if __name__ == "__main__":
    unittest.main()

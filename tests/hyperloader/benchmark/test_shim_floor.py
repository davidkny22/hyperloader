"""Shim-floor primitive harness tests."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[3]
PATH = ROOT / "benches" / "shim_floor.py"
SPEC = importlib.util.spec_from_file_location("shim_floor_benchmark", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ShimFloorHarnessTest(unittest.TestCase):
    """Exercise every exact tier primitive and its input checks."""

    def test_every_tier_primitive_is_timed_and_observable(self) -> None:
        rows = MODULE.measure_operations(2, 1, 2)

        self.assertEqual({row[0] for row in rows}, set(MODULE.METRICS))
        self.assertEqual(len(rows), len(MODULE.METRICS) * 2)
        self.assertTrue(all(row[3] > 0 and row[4] > 0 for row in rows))
        self.assertTrue(all(row[5] != 0 for row in rows))

    def test_nonpositive_counts_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            MODULE.measure_operations(0, 1, 1)


if __name__ == "__main__":
    unittest.main()

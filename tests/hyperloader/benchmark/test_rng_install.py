"""RNG installation microbenchmark harness checks."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[3]
PATH = ROOT / "benches" / "rng_install.py"
SPEC = importlib.util.spec_from_file_location("rng_install_benchmark", PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {PATH}")
rng_install = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rng_install
SPEC.loader.exec_module(rng_install)


class RngInstallHarnessTest(unittest.TestCase):
    """Verify complete metric coverage and observable execution."""

    def test_every_install_component_produces_positive_measurements(self) -> None:
        rows = rng_install.measure_operations(2, 1, 2)

        self.assertEqual({row[0] for row in rows}, set(rng_install.METRICS))
        self.assertEqual(len(rows), len(rng_install.METRICS) * 2)
        self.assertTrue(all(row[3] > 0 and row[4] > 0 for row in rows))
        self.assertTrue(any(row[5] != 0 for row in rows))

    def test_nonpositive_measurement_counts_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            rng_install.measure_operations(0, 1, 1)


if __name__ == "__main__":
    unittest.main()

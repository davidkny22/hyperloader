"""Cost-profile budget and native persistence checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hyperloader import _hyperloader
from hyperloader.profile import profile_budget_bytes


class CostProfileRuntimeTest(unittest.TestCase):
    """Exercise the installed native profile through the Python shell."""

    def test_budget_fraction_is_bounded_by_observed_free_disk(self) -> None:
        self.assertEqual(profile_budget_bytes(0.01, 10_000), 100)
        self.assertEqual(profile_budget_bytes(2.0, 10_000), 10_000)

    def test_native_profile_persists_ema_and_degradation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "costs.bin"
            profile = _hyperloader._CostProfile(10, 16, 0.3)
            profile.observe(4, 100)
            profile.observe(4, 200)
            profile.save(path)

            loaded = _hyperloader._CostProfile.load(path, 10, 16, 0.3)

        self.assertTrue(loaded.degraded)
        self.assertEqual(loaded.payload_bytes, 16)
        self.assertEqual(loaded.estimate(4), 130.0)


if __name__ == "__main__":
    unittest.main()

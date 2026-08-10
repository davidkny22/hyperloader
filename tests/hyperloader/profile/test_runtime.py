"""Cost-profile budget and native persistence checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hyperloader import DataLoader, HyperConfig, _hyperloader
from hyperloader.config import SchedulerConfig
from hyperloader.control import CpuCluster, MachineIdentity
from hyperloader.fingerprint import ContractFingerprint, FingerprintElement
from hyperloader.profile import profile_budget_bytes, profile_cache_path


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

    def test_cache_path_is_dataset_and_machine_specific(self) -> None:
        root = Path("cache")
        dataset = ContractFingerprint((FingerprintElement("dataset.length", 4),))
        machine = MachineIdentity("cpu", (CpuCluster("all", (0,)),), 1024)

        path = profile_cache_path(root, dataset, machine)

        self.assertEqual(path.parent.name, dataset.digest)
        self.assertEqual(path.name, f"{machine.cache_key}.bin")

    def test_loader_reuses_profile_only_for_matching_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = HyperConfig(
                scheduler=SchedulerConfig(profile_cache=Path(directory))
            )
            first = DataLoader(range(4), num_workers=0, config=config)
            first._cost_profile.observe(2, 700)
            first.close()

            matching = DataLoader(range(4), num_workers=0, config=config)
            changed = DataLoader(range(5), num_workers=0, config=config)
            try:
                self.assertEqual(matching._cost_profile.estimate(2), 700.0)
                self.assertIsNone(changed._cost_profile.estimate(2))
                self.assertNotEqual(
                    matching._profile_cache_path, changed._profile_cache_path
                )
            finally:
                matching.close()
                changed.close()

    def test_budget_observes_the_cache_filesystem(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured = root / "not-created"
            config = HyperConfig(scheduler=SchedulerConfig(profile_cache=configured))
            with patch(
                "hyperloader.profile.runtime.shutil.disk_usage",
                return_value=SimpleNamespace(free=10_000),
            ) as disk_usage:
                loader = DataLoader(range(4), num_workers=0, config=config)
                loader.close()

        disk_usage.assert_called_once_with(root)


if __name__ == "__main__":
    unittest.main()

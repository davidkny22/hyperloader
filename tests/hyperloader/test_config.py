"""Tests for the public configuration model."""

import unittest

from hyperloader import AUTO, HyperConfig
from hyperloader.config import (
    DeterminismConfig,
    DistributedConfig,
    ExecutorConfig,
    FactorConfig,
    SchedulerConfig,
)


class ConfigTest(unittest.TestCase):
    """Verify immutable defaults and invalid contract settings."""

    def test_defaults_use_derived_values(self) -> None:
        config = HyperConfig()

        self.assertIs(config.executor.process_ceiling, AUTO)
        self.assertIs(config.scheduler.frontier_depth, AUTO)
        self.assertEqual(repr(AUTO), "auto")
        self.assertEqual(config.factors.f_snap_bytes, 4 * 1024 * 1024)

    def test_unknown_seeded_library_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown libraries: custom"):
            DeterminismConfig(seeded_libs=("torch", "custom"))  # type: ignore[arg-type]

    def test_invalid_distributed_coordinate_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank must be smaller"):
            DistributedConfig(rank=2, world_size=2)

    def test_boolean_process_ceiling_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            ExecutorConfig(process_ceiling=True)

    def test_unknown_worker_death_policy_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "on_worker_death"):
            ExecutorConfig(on_worker_death="ignore")  # type: ignore[arg-type]

    def test_profile_cache_rejects_non_path_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "profile_cache"):
            SchedulerConfig(profile_cache=object())

    def test_profile_ema_factor_cannot_exceed_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "alpha"):
            FactorConfig(alpha=1.1)

    def test_frontier_growth_multiplier_must_make_progress(self) -> None:
        with self.assertRaisesRegex(ValueError, "growth_mult"):
            FactorConfig(growth_mult=1)


if __name__ == "__main__":
    unittest.main()

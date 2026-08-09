"""Tests for the public configuration model."""

import unittest

from hyperloader import AUTO, HyperConfig
from hyperloader.config import DeterminismConfig, DistributedConfig, ExecutorConfig


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


if __name__ == "__main__":
    unittest.main()

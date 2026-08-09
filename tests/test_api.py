"""Constructor contract tests for the public DataLoader."""

import unittest

from hyperloader import AUTO, DataLoader, HyperConfig
from hyperloader.config import ExecutorConfig, MemoryConfig


class DataLoaderValidationTest(unittest.TestCase):
    """Exercise constructor conflicts and hyperloader-specific precedence rules."""

    def test_sampler_and_shuffle_conflict(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            DataLoader([], shuffle=True, sampler=object())

    def test_batch_sampler_conflicts_with_batch_controls(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_sampler is mutually exclusive"):
            DataLoader([], batch_size=2, batch_sampler=object())

    def test_batch_sampler_accepts_torch_default_batch_size(self) -> None:
        loader = DataLoader([], batch_sampler=object())

        self.assertIsNone(loader.batch_size)

    def test_zero_worker_rejects_explicit_prefetch(self) -> None:
        with self.assertRaisesRegex(ValueError, "only be specified"):
            DataLoader([], num_workers=0, prefetch_factor=2)

    def test_delivery_alias_conflict_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicts with in_order"):
            DataLoader([], in_order=False, delivery="in-order")

    def test_seed_alias_conflict_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicts with config.seed"):
            DataLoader([], seed=3, config=HyperConfig(seed=4))

    def test_process_ceiling_conflict_is_rejected(self) -> None:
        config = HyperConfig(executor=ExecutorConfig(process_ceiling=2))

        with self.assertRaisesRegex(ValueError, "process_ceiling"):
            DataLoader([], num_workers=3, config=config)

    def test_delivery_memory_conflict_is_rejected(self) -> None:
        config = HyperConfig(memory=MemoryConfig(delivery_memory="host"))

        with self.assertRaisesRegex(ValueError, "delivery_memory"):
            DataLoader([], pin_memory=True, config=config)

    def test_auto_defaults_resolve_to_identity_contract(self) -> None:
        loader = DataLoader([])

        self.assertIs(loader.num_workers, AUTO)
        self.assertEqual(loader.delivery, "in-order")
        self.assertTrue(loader.persistent_workers)
        self.assertEqual(loader.mode, "native")

    def test_execution_before_planner_wiring_fails_clearly(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "planner is not initialized"):
            iter(DataLoader([]))


if __name__ == "__main__":
    unittest.main()

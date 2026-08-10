"""Loader contract-input fingerprint tests."""

from __future__ import annotations

import unittest

from hyperloader import DataLoader, HyperConfig
from hyperloader.config import (
    DeterminismConfig,
    DistributedConfig,
    MemoryConfig,
    SchedulerConfig,
)


def collate_left(values: list[int]) -> tuple[int, ...]:
    """Provide one stable collation identity."""
    return tuple(values)


def collate_right(values: list[int]) -> list[int]:
    """Provide a distinct collation identity."""
    return values


def init_left(_worker: int) -> None:
    """Provide one iterable sharding initializer identity."""


def init_right(_worker: int) -> None:
    """Provide a distinct iterable sharding initializer identity."""


class IterableFixture:
    """Provide an unsized iterable plan for identity checks."""

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter((1, 2, 3))


class SamplerFixture:
    """Expose stable public sampler configuration."""

    def __init__(self, offset: int) -> None:
        self.offset = offset

    def __iter__(self):
        return iter(range(self.offset, self.offset + 8))

    def __len__(self) -> int:
        return 8


class ContractFingerprintBuilderTest(unittest.TestCase):
    """Exercise every currently reachable result-contract input class."""

    def test_seed_and_map_worker_initializer_are_excluded(self) -> None:
        config = HyperConfig(scheduler=SchedulerConfig(profile_cache="off"))
        first = DataLoader(
            range(8), num_workers=0, seed=1, worker_init_fn=init_left, config=config
        )
        second = DataLoader(
            range(8), num_workers=0, seed=2, worker_init_fn=init_right, config=config
        )

        self.assertEqual(first._fingerprint, second._fingerprint)

    def test_map_contract_controls_change_named_elements(self) -> None:
        base_config = HyperConfig(scheduler=SchedulerConfig(profile_cache="off"))
        base = DataLoader(range(8), batch_size=2, num_workers=0, config=base_config)
        exact = DataLoader(
            range(8),
            batch_size=2,
            num_workers=0,
            config=HyperConfig(
                scheduler=SchedulerConfig(profile_cache="off"),
                determinism=DeterminismConfig(exact_count=True),
            ),
        )
        shaped = DataLoader(
            range(8),
            batch_size=2,
            num_workers=0,
            collate_fn=collate_left,
            config=HyperConfig(
                scheduler=SchedulerConfig(profile_cache="off"),
                memory=MemoryConfig(batch_shape={"dtype": "int64", "shape": [2]}),
                determinism=DeterminismConfig(
                    decoder_pins=("png", "4.0"),
                    seeded_libs=("torch", "random", "numpy"),
                ),
                distributed=DistributedConfig(world_size=2, rank=0),
            ),
        )
        values = _values(shaped)

        self.assertNotEqual(base._fingerprint.digest, exact._fingerprint.digest)
        self.assertEqual(values["placement.B_g"], 4)
        self.assertEqual(values["batch_shape"]["source"], "declared")
        self.assertIn("collate_left", values["collate.identity"]["qualname"])
        self.assertEqual(values["seeded_libs"], ["torch", "random", "numpy"])
        self.assertEqual(values["decoder_pins"], ["png", "4.0"])

    def test_collate_delivery_and_drop_controls_are_distinct(self) -> None:
        config = HyperConfig(scheduler=SchedulerConfig(profile_cache="off"))
        left = DataLoader(
            range(8),
            num_workers=0,
            collate_fn=collate_left,
            drop_last=False,
            config=config,
        )
        right = DataLoader(
            range(8),
            num_workers=0,
            collate_fn=collate_right,
            drop_last=True,
            in_order=False,
            config=config,
        )

        self.assertNotEqual(left._fingerprint.digest, right._fingerprint.digest)
        self.assertEqual(_values(right)["delivery"], "on-completion")
        self.assertTrue(_values(right)["drop_last"])

        shuffled = DataLoader(
            range(8),
            num_workers=0,
            shuffle=True,
            collate_fn=collate_left,
            config=config,
        )
        self.assertNotEqual(left._fingerprint.digest, shuffled._fingerprint.digest)
        self.assertTrue(_values(shuffled)["sampler.shuffle"])

    def test_user_sampler_identity_uses_the_zero_global_batch_sentinel(self) -> None:
        config = HyperConfig(scheduler=SchedulerConfig(profile_cache="off"))
        first = DataLoader(
            range(16),
            batch_size=2,
            num_workers=0,
            sampler=SamplerFixture(0),
            config=config,
        )
        second = DataLoader(
            range(16),
            batch_size=2,
            num_workers=0,
            sampler=SamplerFixture(1),
            config=config,
        )

        self.assertEqual(_values(first)["placement.B_g"], 0)
        self.assertNotEqual(first._fingerprint.digest, second._fingerprint.digest)

    def test_iterable_contract_adds_lanes_world_batch_and_initializer(self) -> None:
        first = DataLoader(
            IterableFixture(),
            batch_size=3,
            num_workers=2,
            worker_init_fn=init_left,
            config=HyperConfig(
                scheduler=SchedulerConfig(profile_cache="off"),
                distributed=DistributedConfig(world_size=4, rank=1),
            ),
        )
        second = DataLoader(
            IterableFixture(),
            batch_size=3,
            num_workers=2,
            worker_init_fn=init_right,
            config=HyperConfig(
                scheduler=SchedulerConfig(profile_cache="off"),
                distributed=DistributedConfig(world_size=4, rank=1),
            ),
        )
        values = _values(first)

        self.assertEqual(values["iterable.L"], 2)
        self.assertEqual(values["iterable.W"], 4)
        self.assertEqual(values["iterable.batch_size"], 3)
        self.assertNotEqual(first._fingerprint.digest, second._fingerprint.digest)


def _values(loader: DataLoader) -> dict[str, object]:
    return {element.path: element.value for element in loader._fingerprint.elements}


if __name__ == "__main__":
    unittest.main()

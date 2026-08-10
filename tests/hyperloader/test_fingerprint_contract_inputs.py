"""Installed contract-control fingerprint mutation checks."""

from __future__ import annotations

import unittest

from hyperloader import DataLoader, HyperConfig
from hyperloader.config import (
    DeterminismConfig,
    DistributedConfig,
    MemoryConfig,
    SchedulerConfig,
)

from .fingerprint_invalidation_support import (
    IterableValues,
    OffsetSampler,
    collate_list,
    collate_tuple,
    init_left,
    init_right,
)


class ContractInputFingerprintTest(unittest.TestCase):
    """Prove every configurable contract-control group changes identity."""

    def test_every_contract_control_changes_its_named_element(self) -> None:
        off = SchedulerConfig(profile_cache="off")
        base = DataLoader(
            range(8), batch_size=2, num_workers=0, config=HyperConfig(scheduler=off)
        )
        cases = [
            DataLoader(
                range(8), batch_size=4, num_workers=0, config=HyperConfig(scheduler=off)
            ),
            DataLoader(
                range(8),
                batch_size=2,
                num_workers=0,
                sampler=OffsetSampler(8, 0),
                config=HyperConfig(scheduler=off),
            ),
            DataLoader(
                range(8),
                batch_size=2,
                num_workers=0,
                drop_last=True,
                in_order=False,
                collate_fn=collate_tuple,
                config=HyperConfig(scheduler=off),
            ),
            DataLoader(
                range(8),
                batch_size=2,
                num_workers=0,
                config=HyperConfig(
                    scheduler=off,
                    determinism=DeterminismConfig(
                        exact_count=True,
                        seeded_libs=("torch",),
                    ),
                ),
            ),
            DataLoader(
                range(8),
                batch_size=2,
                num_workers=0,
                config=HyperConfig(
                    scheduler=off,
                    memory=MemoryConfig(batch_shape={"dtype": "int64", "shape": [2]}),
                ),
            ),
        ]
        iterable_left = DataLoader(
            IterableValues(),
            batch_size=2,
            num_workers=2,
            worker_init_fn=init_left,
            config=HyperConfig(
                scheduler=off, distributed=DistributedConfig(world_size=2, rank=0)
            ),
        )
        iterable_right = DataLoader(
            IterableValues(),
            batch_size=3,
            num_workers=3,
            worker_init_fn=init_right,
            config=HyperConfig(
                scheduler=off, distributed=DistributedConfig(world_size=3, rank=0)
            ),
        )
        try:
            for changed in cases:
                self.assertNotEqual(
                    base._fingerprint.digest, changed._fingerprint.digest
                )
            self.assertNotEqual(
                iterable_left._fingerprint.digest, iterable_right._fingerprint.digest
            )
            self.assertNotEqual(
                self._fingerprint(range(8), off, collate_tuple),
                self._fingerprint(range(8), off, collate_list),
            )
        finally:
            base.close()
            for changed in cases:
                changed.close()
            iterable_left.close()
            iterable_right.close()

    @staticmethod
    def _fingerprint(
        dataset: object, scheduler: SchedulerConfig, collate_fn: object
    ) -> str:
        loader = DataLoader(
            dataset,
            num_workers=0,
            collate_fn=collate_fn,
            config=HyperConfig(scheduler=scheduler),
        )
        try:
            return loader._fingerprint.digest
        finally:
            loader.close()


if __name__ == "__main__":
    unittest.main()

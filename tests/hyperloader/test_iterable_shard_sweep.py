"""Installed public gate for stateful-source iterable sharding."""

from __future__ import annotations

import json
import os
import unittest
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest import mock

import hyperloader
from hyperloader import DataLoader, HyperConfig, _hyperloader
from hyperloader.config import DistributedConfig
from hyperloader.iterable import sharding
from torch.utils.data import get_worker_info


class ProtocolSource:
    """Partition sample identifiers by supplied rank and lane coordinates."""

    def __init__(self, length: int = 97) -> None:
        self.length = length
        self.assignment: tuple[int, int, int, int] | None = None

    def shard(self, rank: int, world_size: int, lane: int, lanes: int) -> None:
        self.assignment = rank, world_size, lane, lanes

    def __iter__(self):
        if self.assignment is None:
            raise RuntimeError("source shard was not applied")
        rank, world_size, lane, lanes = self.assignment
        slot = rank * lanes + lane
        for sample in range(slot, self.length, world_size * lanes):
            yield sample, rank, world_size, lane, lanes


class InvalidShardSource:
    """Expose a malformed protocol attribute."""

    shard = 1

    def __iter__(self):
        yield 0


def _assert_shard_precedes_initialization(lane: int) -> None:
    info = get_worker_info()
    if info is None:
        raise AssertionError("logical worker identity is absent")
    assignment = info.dataset.assignment
    if assignment is None:
        raise AssertionError("source shard must run before worker initialization")
    if assignment[2] != lane:
        raise AssertionError("source shard lane does not match worker identity")


def _rows(batch: Any) -> list[tuple[int, int, int, int, int]]:
    return [
        tuple(int(value) for value in row)
        for row in zip(*(column.tolist() for column in batch), strict=True)
    ]


def _run_topology(world_size: int, lanes: int) -> list[tuple[int, ...]]:
    records = []
    for rank in range(world_size):
        loader = DataLoader(
            ProtocolSource(),
            batch_size=3,
            num_workers=lanes,
            seed=827,
            config=HyperConfig(
                distributed=DistributedConfig(rank=rank, world_size=world_size)
            ),
        )
        try:
            for batch in loader:
                records.extend(_rows(batch))
        finally:
            loader.close()
    return records


def _ignore_rank(dataset: Any, topology: Any, lane: int, lanes: int) -> Any:
    replacement = dataset.shard(0, topology.world_size, lane, lanes)
    return dataset if replacement is None else replacement


class IterableShardSweepGate(unittest.TestCase):
    """Prove protocol partitions across metamorphic rank and lane topologies."""

    def test_shards_are_covering_disjoint_deterministic_and_lane_stable(self) -> None:
        expected_root = os.environ.get("HYPERLOADER_EXPECTED_INSTALL_ROOT")
        if expected_root is not None:
            root = Path(expected_root).resolve()
            self.assertTrue(Path(hyperloader.__file__).resolve().is_relative_to(root))
            self.assertTrue(Path(_hyperloader.__file__).resolve().is_relative_to(root))

        mutation = (
            mock.patch.object(sharding, "apply_source_shard", _ignore_rank)
            if os.environ.get("HYPERLOADER_ITERABLE_SHARD_MUTATION") == "ignore-rank"
            else nullcontext()
        )
        evidence = []
        with mutation:
            for world_size, lanes in ((1, 1), (1, 4), (2, 2), (4, 1)):
                with self.subTest(world_size=world_size, lanes=lanes):
                    first = _run_topology(world_size, lanes)
                    second = _run_topology(world_size, lanes)
                    self.assertEqual(first, second)

                    owners: dict[tuple[int, int], list[int]] = defaultdict(list)
                    for sample, rank, observed_world, lane, observed_lanes in first:
                        self.assertEqual(observed_world, world_size)
                        self.assertEqual(observed_lanes, lanes)
                        slot = sample % (world_size * lanes)
                        self.assertEqual((rank, lane), divmod(slot, lanes))
                        owners[(rank, lane)].append(sample)

                    flattened = [
                        sample for samples in owners.values() for sample in samples
                    ]
                    self.assertEqual(sorted(flattened), list(range(97)))
                    self.assertEqual(len(flattened), len(set(flattened)))
                    evidence.append(
                        {
                            "lanes": lanes,
                            "owners": len(owners),
                            "samples": len(flattened),
                            "world_size": world_size,
                        }
                    )

        metrics_path = os.environ.get("HYPERLOADER_ITERABLE_SHARD_METRICS")
        if metrics_path is not None:
            Path(metrics_path).write_text(
                json.dumps({"topologies": evidence}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def test_noncallable_shard_is_rejected(self) -> None:
        loader = DataLoader(InvalidShardSource(), batch_size=1, num_workers=1)
        try:
            with self.assertRaisesRegex(TypeError, "source shard must be callable"):
                iter(loader)
        finally:
            loader.close()

    def test_shard_precedes_worker_initialization(self) -> None:
        loader = DataLoader(
            ProtocolSource(length=8),
            batch_size=2,
            num_workers=2,
            worker_init_fn=_assert_shard_precedes_initialization,
        )
        try:
            self.assertEqual(len(list(loader)), 4)
        finally:
            loader.close()


if __name__ == "__main__":
    unittest.main()

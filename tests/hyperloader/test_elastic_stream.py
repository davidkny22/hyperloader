"""Installed public gate for topology-independent map continuation."""

from __future__ import annotations

import json
import os
import unittest
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest import mock

import hyperloader
from hyperloader import DataLoader, HyperConfig, _hyperloader
from hyperloader.config import DistributedConfig, SchedulerConfig
from hyperloader.distributed import MapPlacement


class CoordinateDataset:
    """Return the selected index and its engine-owned random stream."""

    def __init__(self, length: int = 101) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[int, int]:
        return index, hyperloader.rng("random").getrandbits(63)


def _config(world_size: int, rank: int) -> HyperConfig:
    return HyperConfig(
        distributed=DistributedConfig(world_size=world_size, rank=rank),
        scheduler=SchedulerConfig(profile_cache="off"),
    )


def _loader(
    *,
    batch_size: int,
    world_size: int,
    rank: int,
    seed: int,
) -> DataLoader:
    return DataLoader(
        CoordinateDataset(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        thread_safe=True,
        seed=seed,
        config=_config(world_size, rank),
    )


def _rows(batch: Any) -> list[tuple[int, int]]:
    return [
        (int(index), int(bits))
        for index, bits in zip(
            batch[0].tolist(),
            batch[1].tolist(),
            strict=True,
        )
    ]


def _stream(loader: DataLoader) -> list[list[tuple[int, int]]]:
    return [_rows(batch) for batch in loader]


def _rank_local_coordinate(placement: MapPlacement, position: int) -> int:
    if not 0 <= position < placement.length:
        raise IndexError("rank-local sampler position is outside the placement")
    return position


class ElasticStreamGate(unittest.TestCase):
    """Prove exact global continuation across admissible world sizes."""

    def test_checkpoint_continuation_is_topology_independent(self) -> None:
        expected_root = os.environ.get("HYPERLOADER_EXPECTED_INSTALL_ROOT")
        if expected_root is not None:
            root = Path(expected_root).resolve()
            self.assertTrue(Path(hyperloader.__file__).resolve().is_relative_to(root))
            self.assertTrue(Path(_hyperloader.__file__).resolve().is_relative_to(root))

        source = _loader(batch_size=48, world_size=1, rank=0, seed=613)
        iterator = iter(source)
        try:
            first_batch = _rows(next(iterator))
            state = source.state_dict()
            expected = [_rows(batch) for batch in iterator]
        finally:
            source.close()

        self.assertEqual(len(first_batch), 48)
        self.assertEqual(state["B_g"], 48)
        self.assertEqual(state["cursor"], 1)
        self.assertEqual([len(batch) for batch in expected], [48, 48])

        mutation = (
            mock.patch.object(MapPlacement, "coordinate", _rank_local_coordinate)
            if os.environ.get("HYPERLOADER_ELASTIC_MUTATION")
            == "local-coordinate"
            else nullcontext()
        )
        evidence = []
        with mutation:
            for world_size in (1, 2, 8, 48):
                with self.subTest(world_size=world_size):
                    per_rank_batch = 48 // world_size
                    rank_streams = []
                    for rank in range(world_size):
                        resumed = _loader(
                            batch_size=per_rank_batch,
                            world_size=world_size,
                            rank=rank,
                            seed=999,
                        )
                        try:
                            resumed.load_state_dict(state)
                            rank_streams.append(_stream(resumed))
                        finally:
                            resumed.close()

                    actual = [
                        [
                            row
                            for rank in range(world_size)
                            for row in rank_streams[rank][batch]
                        ]
                        for batch in range(len(expected))
                    ]
                    self.assertEqual(actual, expected)
                    self.assertEqual([len(stream) for stream in rank_streams], [2] * world_size)
                    evidence.append(
                        {
                            "global_batch": 48,
                            "per_rank_batch": per_rank_batch,
                            "remaining_global_batches": len(actual),
                            "remaining_rows": sum(len(batch) for batch in actual),
                            "world_size": world_size,
                        }
                    )

        metrics_path = os.environ.get("HYPERLOADER_ELASTIC_METRICS")
        if metrics_path is not None:
            Path(metrics_path).write_text(
                json.dumps({"topologies": evidence}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def test_invalid_world_and_rank_batch_name_the_requirement(self) -> None:
        source = _loader(batch_size=48, world_size=1, rank=0, seed=613)
        try:
            state = source.state_dict()
        finally:
            source.close()

        nonintegral = _loader(batch_size=9, world_size=5, rank=0, seed=999)
        wrong_batch = _loader(batch_size=12, world_size=8, rank=0, seed=999)
        try:
            with self.assertRaisesRegex(ValueError, "batch_size=48/5.*not an integer"):
                nonintegral.load_state_dict(state)
            with self.assertRaisesRegex(ValueError, "batch_size=6, not 12"):
                wrong_batch.load_state_dict(state)
        finally:
            nonintegral.close()
            wrong_batch.close()


if __name__ == "__main__":
    unittest.main()

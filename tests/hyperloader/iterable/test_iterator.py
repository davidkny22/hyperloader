"""Logical iterable lane construction and delivery."""

from __future__ import annotations

import itertools
import random
import unittest

import numpy as np
import torch
from hyperloader import DataLoader
from hyperloader.iterable import iterable_coordinate
from hyperloader.rng import rng
from torch.utils.data import get_worker_info


class ShardedIterable:
    """Use the logical worker view to partition one finite range."""

    def __init__(self, stop: int = 10) -> None:
        self.stop = stop

    def __iter__(self):
        info = get_worker_info()
        if info is None:
            return iter(range(self.stop))
        return iter(range(info.id, self.stop, info.num_workers))


class InfiniteIterable:
    """Yield the lane identity forever."""

    def __iter__(self):
        info = get_worker_info()
        identity = -1 if info is None else info.id
        return itertools.repeat(identity)


class MutableIterable:
    """Expose initializer mutation from each lane-owned copy."""

    def __init__(self) -> None:
        self.offset = 0

    def __iter__(self):
        yield self.offset


class UnpicklableIterable:
    """Carry a value that the standard pickle protocol rejects."""

    def __init__(self) -> None:
        self.callback = lambda: None

    def __iter__(self):
        yield 1


class SeededIterable:
    """Expose logical identity and every seeded sample RNG surface."""

    def __init__(self, count: int = 2) -> None:
        self.count = count
        self.worker_infos = []

    def __iter__(self):
        for _ in range(self.count):
            info = get_worker_info()
            if info is None:
                raise AssertionError("logical worker identity is absent")
            self.worker_infos.append(info)
            yield (
                info.id,
                info.num_workers,
                int(info.seed is not None),
                id(info),
                random.getrandbits(32),
                int(np.random.randint(0, 1 << 31)),
                int(torch.randint(0, 1 << 31, ()).item()),
                rng("random").getrandbits(32),
            )


def _set_lane_offset(lane: int) -> None:
    info = get_worker_info()
    if info is None:
        raise AssertionError("logical worker identity is absent")
    info.dataset.offset = lane * 10


def _batches(loader: DataLoader) -> list[tuple[int, ...]]:
    return [tuple(int(value) for value in batch.tolist()) for batch in loader]


def _records(loader: DataLoader) -> list[tuple[int, ...]]:
    records = []
    for batch in loader:
        records.extend(
            tuple(int(value) for value in row)
            for row in zip(
                *(column.tolist() for column in batch),
                strict=True,
            )
        )
    return records


class IterableLaneTest(unittest.TestCase):
    """Keep logical lane behavior independent of execution timing."""

    def test_round_robin_lane_whole_batches_and_exhaustion(self) -> None:
        loader = DataLoader(
            ShardedIterable(),
            batch_size=2,
            num_workers=3,
        )
        try:
            self.assertEqual(
                _batches(loader),
                [(0, 3), (1, 4), (2, 5), (6, 9), (7,), (8,)],
            )
        finally:
            loader.close()

    def test_drop_last_applies_per_lane(self) -> None:
        loader = DataLoader(
            ShardedIterable(),
            batch_size=2,
            num_workers=3,
            drop_last=True,
        )
        try:
            self.assertEqual(_batches(loader), [(0, 3), (1, 4), (2, 5), (6, 9)])
        finally:
            loader.close()

    def test_initializer_mutates_independent_lane_copies(self) -> None:
        dataset = MutableIterable()
        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=3,
            worker_init_fn=_set_lane_offset,
        )
        try:
            self.assertEqual(_batches(loader), [(0,), (10,), (20,)])
            self.assertEqual(dataset.offset, 0)
        finally:
            loader.close()

    def test_infinite_lanes_remain_fixed_and_round_robin(self) -> None:
        loader = DataLoader(InfiniteIterable(), batch_size=2, num_workers=2)
        iterator = iter(loader)
        try:
            self.assertEqual(
                [
                    tuple(int(value) for value in next(iterator).tolist())
                    for _ in range(4)
                ],
                [(0, 0), (1, 1), (0, 0), (1, 1)],
            )
        finally:
            loader.close()

    def test_unpicklable_multi_lane_source_names_the_single_lane_fallback(self) -> None:
        with self.assertRaisesRegex(TypeError, "num_workers=0 for the L=1 fallback"):
            DataLoader(UnpicklableIterable(), batch_size=1, num_workers=2)

        loader = DataLoader(UnpicklableIterable(), batch_size=1, num_workers=0)
        try:
            self.assertEqual(_batches(loader), [(1,)])
        finally:
            loader.close()

    def test_abandoned_iterable_replays_the_same_epoch(self) -> None:
        loader = DataLoader(ShardedIterable(6), batch_size=1, num_workers=2)
        first = iter(loader)
        try:
            prefix = tuple(int(value) for value in next(first).tolist())
            replay = iter(loader)
            self.assertEqual(
                tuple(int(value) for value in next(replay).tolist()), prefix
            )
        finally:
            loader.close()

    def test_sample_rng_and_worker_identity_are_coordinate_bound(self) -> None:
        left = DataLoader(SeededIterable(), batch_size=2, num_workers=2, seed=733)
        right = DataLoader(SeededIterable(), batch_size=2, num_workers=2, seed=733)
        try:
            random_state = random.getstate()
            numpy_state = np.random.get_state()
            torch_state = torch.default_generator.get_state()
            left_records = _records(left)
            self.assertEqual(random.getstate(), random_state)
            current_numpy = np.random.get_state()
            self.assertEqual(current_numpy[0], numpy_state[0])
            self.assertTrue(np.array_equal(current_numpy[1], numpy_state[1]))
            self.assertEqual(current_numpy[2:], numpy_state[2:])
            self.assertTrue(
                torch.equal(torch.default_generator.get_state(), torch_state)
            )

            right_records = _records(right)
            left_values = [record[:3] + record[4:] for record in left_records]
            right_values = [record[:3] + record[4:] for record in right_records]
            self.assertEqual(left_values, right_values)
            self.assertEqual([record[0] for record in left_records], [0, 0, 1, 1])
            self.assertTrue(all(record[1:3] == (2, 1) for record in left_records))
            self.assertEqual(len({record[3] for record in left_records}), 4)
        finally:
            left.close()
            right.close()

    def test_iterable_coordinate_validates_each_packed_field(self) -> None:
        self.assertEqual(iterable_coordinate(3, 5, 7), (3 << 52) | (5 << 40) | 7)
        for args, name in (
            ((1 << 12, 0, 0), "rank"),
            ((0, 1 << 12, 0), "lane"),
            ((0, 0, 1 << 40), "arrival"),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                iterable_coordinate(*args)


if __name__ == "__main__":
    unittest.main()

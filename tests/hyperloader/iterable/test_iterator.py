"""Logical iterable lane construction and delivery."""

from __future__ import annotations

import itertools
import unittest

from hyperloader import DataLoader
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


def _set_lane_offset(lane: int) -> None:
    info = get_worker_info()
    if info is None:
        raise AssertionError("logical worker identity is absent")
    info.dataset.offset = lane * 10


def _batches(loader: DataLoader) -> list[tuple[int, ...]]:
    return [tuple(int(value) for value in batch.tolist()) for batch in loader]


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


if __name__ == "__main__":
    unittest.main()

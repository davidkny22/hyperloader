"""Map placement and elastic checkpoint restoration."""

from __future__ import annotations

import random
import unittest

from hyperloader import DataLoader, HyperConfig, _hyperloader
from hyperloader.config import DistributedConfig, SchedulerConfig
from hyperloader.distributed import MapPlacement
from hyperloader.planner import BlackBoxPlan


class RangeDataset:
    """Return deterministic indices from a sized map source."""

    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return index


class RandomDataset(RangeDataset):
    """Expose the coordinate-bound Python random stream beside each index."""

    def __getitem__(self, index: int) -> tuple[int, int]:
        return index, random.getrandbits(32)


def _config(world_size: int, rank: int) -> HyperConfig:
    return HyperConfig(
        distributed=DistributedConfig(world_size=world_size, rank=rank),
        scheduler=SchedulerConfig(frontier_depth=8, profile_cache="off"),
    )


def _loader(
    dataset: object,
    *,
    batch_size: int,
    world_size: int,
    rank: int,
    thread_safe: bool = False,
    seed: int = 701,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        thread_safe=thread_safe,
        seed=seed,
        config=_config(world_size, rank),
    )


def _rows(batch: object) -> list[tuple[int, ...]]:
    columns = batch  # type: ignore[assignment]
    return [
        tuple(int(value) for value in row)
        for row in zip(*(column.tolist() for column in columns), strict=True)
    ]


class MapPlacementTest(unittest.TestCase):
    """Preserve one global stream under rank-local execution."""

    def test_local_coordinates_match_the_native_placement_contract(self) -> None:
        for drop_last, exact_count in ((False, False), (True, False), (False, True)):
            for rank in range(4):
                with self.subTest(
                    drop_last=drop_last, exact_count=exact_count, rank=rank
                ):
                    plan = BlackBoxPlan(length=13, shuffle=True)
                    placement = MapPlacement(
                        dataset_length=13,
                        batch_size=2,
                        rank=rank,
                        world_size=4,
                        drop_last=drop_last,
                        exact_count=exact_count,
                        enabled=True,
                    )
                    actual = [
                        (
                            placement.coordinate(position),
                            placement.index(plan, 17, 3, position),
                        )
                        for position in range(placement.length)
                    ]
                    expected = _hyperloader._rank_placements(
                        17,
                        3,
                        13,
                        2,
                        4,
                        rank,
                        drop_last,
                        exact_count,
                    )
                    self.assertEqual(actual, expected)

    def test_process_restore_preserves_indices_and_rng_across_world_size(self) -> None:
        dataset = RandomDataset(13)
        source = _loader(dataset, batch_size=4, world_size=1, rank=0)
        iterator = iter(source)
        try:
            next(iterator)
            state = source.state_dict()
            expected = [_rows(batch) for batch in iterator]
        finally:
            source.close()

        rank_batches = []
        for rank in range(2):
            resumed = _loader(
                dataset,
                batch_size=2,
                world_size=2,
                rank=rank,
                seed=999,
            )
            try:
                resumed.load_state_dict(state)
                rank_batches.append([_rows(batch) for batch in resumed])
            finally:
                resumed.close()

        actual = [
            rank_batches[0][batch] + rank_batches[1][batch]
            for batch in range(len(expected))
        ]
        self.assertEqual(state["B_g"], 4)
        self.assertEqual(actual, expected)

    def test_thread_restore_preserves_global_batch_order(self) -> None:
        dataset = RangeDataset(13)
        source = _loader(
            dataset,
            batch_size=4,
            world_size=1,
            rank=0,
            thread_safe=True,
        )
        iterator = iter(source)
        try:
            next(iterator)
            state = source.state_dict()
            expected = [
                tuple(int(value) for value in batch.tolist()) for batch in iterator
            ]
        finally:
            source.close()

        rank_batches = []
        for rank in range(2):
            resumed = _loader(
                dataset,
                batch_size=2,
                world_size=2,
                rank=rank,
                thread_safe=True,
                seed=999,
            )
            try:
                resumed.load_state_dict(state)
                rank_batches.append(
                    [tuple(int(value) for value in batch.tolist()) for batch in resumed]
                )
            finally:
                resumed.close()
        actual = [
            rank_batches[0][batch] + rank_batches[1][batch]
            for batch in range(len(expected))
        ]
        self.assertEqual(actual, expected)

    def test_restore_names_nonintegral_and_required_rank_batch_sizes(self) -> None:
        source = _loader(
            RangeDataset(13),
            batch_size=4,
            world_size=2,
            rank=0,
            thread_safe=True,
        )
        try:
            state = source.state_dict()
        finally:
            source.close()

        nonintegral = _loader(
            RangeDataset(13),
            batch_size=2,
            world_size=3,
            rank=0,
            thread_safe=True,
        )
        wrong_batch = _loader(
            RangeDataset(13),
            batch_size=4,
            world_size=4,
            rank=0,
            thread_safe=True,
        )
        try:
            with self.assertRaisesRegex(ValueError, "batch_size=8/3.*not an integer"):
                nonintegral.load_state_dict(state)
            with self.assertRaisesRegex(ValueError, "batch_size=2, not 4"):
                wrong_batch.load_state_dict(state)
        finally:
            nonintegral.close()
            wrong_batch.close()


if __name__ == "__main__":
    unittest.main()

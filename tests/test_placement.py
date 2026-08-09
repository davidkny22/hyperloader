"""Metamorphic and differential tests for native placement."""

import itertools
import math
import unittest
from collections import Counter

from hyperloader import _hyperloader
from placement_reference import rank_placements


class PlacementTest(unittest.TestCase):
    """Verify coverage, disjointness, padding, and exact-count tails."""

    def test_native_placement_matches_reference_matrix(self) -> None:
        datasets = (0, 1, 3, 17, 127, 1_000)
        topologies = ((1, 1), (2, 3), (8, 2), (48, 1))
        modes = ((False, False), (True, False), (False, True))

        for dataset_len, topology, mode in itertools.product(datasets, topologies, modes):
            world_size, batch_size = topology
            drop_last, exact_count = mode
            for rank in range(world_size):
                arguments = (
                    11,
                    3,
                    dataset_len,
                    batch_size,
                    world_size,
                    rank,
                    drop_last,
                    exact_count,
                )
                with self.subTest(arguments=arguments):
                    self.assertEqual(
                        _hyperloader._rank_placements(*arguments),
                        rank_placements(*arguments),
                    )

    def test_default_padding_is_disjoint_by_position_and_equal_by_rank(self) -> None:
        dataset_len, batch_size, world_size = 5, 4, 8
        ranks = [
            _hyperloader._rank_placements(7, 0, dataset_len, batch_size, world_size, rank)
            for rank in range(world_size)
        ]
        positions = [position for rank in ranks for position, _ in rank]
        indices = [index for rank in ranks for _, index in rank]

        self.assertTrue(all(len(rank) == batch_size for rank in ranks))
        self.assertEqual(sorted(positions), list(range(batch_size * world_size)))
        self.assertEqual(max(Counter(indices).values()), math.ceil(len(positions) / dataset_len))

    def test_exact_count_is_exhaustive_and_disjoint(self) -> None:
        dataset_len, batch_size, world_size = 103, 4, 8
        ranks = [
            _hyperloader._rank_placements(
                7, 0, dataset_len, batch_size, world_size, rank, False, True
            )
            for rank in range(world_size)
        ]
        positions = [position for rank in ranks for position, _ in rank]
        indices = [index for rank in ranks for _, index in rank]

        self.assertEqual(sorted(positions), list(range(dataset_len)))
        self.assertEqual(len(set(indices)), dataset_len)

    def test_drop_last_discards_only_the_global_tail(self) -> None:
        dataset_len, batch_size, world_size = 103, 4, 8
        positions = [
            position
            for rank in range(world_size)
            for position, _ in _hyperloader._rank_placements(
                7, 0, dataset_len, batch_size, world_size, rank, True, False
            )
        ]

        self.assertEqual(sorted(positions), list(range(96)))

    def test_invalid_topology_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "ZeroWorldSize"):
            _hyperloader._rank_placements(0, 0, 10, 1, 0, 0)


if __name__ == "__main__":
    unittest.main()

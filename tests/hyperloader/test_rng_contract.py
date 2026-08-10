"""Cross-language tests for native sample RNG derivation."""

import itertools
import struct
import unittest

from hyperloader import _hyperloader
from .rng_reference import (
    block,
    feistel_permute,
    materialized_permutation,
    permutation_index,
    philox4x32_10,
    mt19937_state,
    sample_torch_seed,
)


class RngContractTest(unittest.TestCase):
    """Compare Rust output with a clear independent Python implementation."""

    def test_random123_zero_known_answer(self) -> None:
        self.assertEqual(
            philox4x32_10((0, 0, 0, 0), (0, 0)),
            (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8),
        )

    def test_native_blocks_match_boundary_matrix(self) -> None:
        root_seeds = (0, 1, (1 << 64) - 1)
        epochs = (0, 1, (1 << 32) - 1)
        coordinates = (0, 1, (1 << 32) - 1, 1 << 32, (1 << 64) - 1)
        draws = (0, 1, 2, (1 << 32) - 1)
        streams = (0, 1, 4, 5, 6, 7, 8)

        for arguments in itertools.product(
            root_seeds, epochs, coordinates, draws, streams
        ):
            with self.subTest(arguments=arguments):
                self.assertEqual(_hyperloader._rng_block(*arguments), block(*arguments))

    def test_sample_rng_states_match_reference(self) -> None:
        cases = ((0, 0, 0), (1, 7, 19), ((1 << 64) - 1, (1 << 32) - 1, (1 << 64) - 1))

        for arguments in cases:
            with self.subTest(arguments=arguments):
                torch_seed, random_bytes, numpy_bytes = _hyperloader._sample_rng_states(
                    *arguments
                )
                random_state = struct.unpack("=625I", random_bytes)
                numpy_state = struct.unpack("=624I", numpy_bytes)

                self.assertEqual(torch_seed, sample_torch_seed(*arguments))
                self.assertEqual(random_state[:-1], mt19937_state(*arguments, 7))
                self.assertEqual(random_state[-1], 624)
                self.assertEqual(numpy_state, mt19937_state(*arguments, 8))

    def test_named_streams_are_separated(self) -> None:
        blocks = {
            _hyperloader._rng_block(11, 3, 29, 2, stream)
            for stream in (0, 1, 4, 5, 6, 7, 8)
        }

        self.assertEqual(len(blocks), 7)

    def test_large_permutations_match_reference(self) -> None:
        domains = (1 << 17, (1 << 17) + 1, 300_000, 1 << 20, 1_000_000_007)
        seeds = ((0, 0), (1, 7), ((1 << 64) - 1, (1 << 32) - 1))

        for domain, seed_epoch in itertools.product(domains, seeds):
            positions = (0, 1, domain // 2, domain - 2, domain - 1)
            for position in positions:
                arguments = (*seed_epoch, domain, position)
                with self.subTest(arguments=arguments):
                    self.assertEqual(
                        _hyperloader._feistel_permute(*arguments),
                        feistel_permute(*arguments),
                    )

    def test_feistel_rejects_small_domain(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 131072"):
            _hyperloader._feistel_permute(0, 0, (1 << 17) - 1, 0)

    def test_materialized_permutations_match_reference(self) -> None:
        domains = (0, 1, 2, 3, 31, 65_536, (1 << 17) - 1)
        seeds = ((0, 0), (1, 7), ((1 << 64) - 1, (1 << 32) - 1))

        for domain, seed_epoch in itertools.product(domains, seeds):
            with self.subTest(domain=domain, seed_epoch=seed_epoch):
                expected, _ = materialized_permutation(*seed_epoch, domain)
                self.assertEqual(
                    _hyperloader._materialized_permutation(*seed_epoch, domain),
                    expected,
                )

    def test_unified_index_matches_reference_across_threshold(self) -> None:
        for domain in (1, 2, 3, 65_536, (1 << 17) - 1, 1 << 17, (1 << 17) + 1, 300_000):
            for position in {0, domain // 2, domain - 1}:
                arguments = (19, 2, domain, position)
                with self.subTest(arguments=arguments):
                    self.assertEqual(
                        _hyperloader._permutation_index(*arguments),
                        permutation_index(*arguments),
                    )

    def test_rejection_attempt_advances_draw_ordinal(self) -> None:
        domain = (1 << 17) - 1
        found_rejection = False
        for root_seed in range(16):
            _, draws = materialized_permutation(root_seed, 0, domain)
            if draws > domain - 1:
                found_rejection = True
                self.assertEqual(
                    _hyperloader._materialized_permutation(root_seed, 0, domain),
                    materialized_permutation(root_seed, 0, domain)[0],
                )
                break
        self.assertTrue(found_rejection, "the test seed set must exercise rejection")


if __name__ == "__main__":
    unittest.main()

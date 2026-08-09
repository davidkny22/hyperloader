"""Cross-language tests for native sample RNG derivation."""

import itertools
import unittest

from hyperloader import _hyperloader
from rng_reference import block, feistel_permute, philox4x32_10, sample_seed_words


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
        streams = (0, 1, 4, 5, 6)

        for arguments in itertools.product(
            root_seeds, epochs, coordinates, draws, streams
        ):
            with self.subTest(arguments=arguments):
                self.assertEqual(_hyperloader._rng_block(*arguments), block(*arguments))

    def test_reserved_seed_words_match_reference(self) -> None:
        cases = ((0, 0, 0), (1, 7, 19), ((1 << 64) - 1, (1 << 32) - 1, (1 << 64) - 1))

        for arguments in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(
                    _hyperloader._sample_seed_words(*arguments),
                    sample_seed_words(*arguments),
                )

    def test_named_streams_are_separated(self) -> None:
        blocks = {_hyperloader._rng_block(11, 3, 29, 2, stream) for stream in (0, 1, 4, 5, 6)}

        self.assertEqual(len(blocks), 5)

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


if __name__ == "__main__":
    unittest.main()

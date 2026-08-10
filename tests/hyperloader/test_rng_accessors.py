"""Public per-sample generator contract tests."""

from __future__ import annotations

import unittest

from hyperloader import DataLoader, _hyperloader, rng
from hyperloader.process.numpy_surface import _splitmix64
from hyperloader.process.random_surface import PhiloxRandom
from hyperloader.rng import _cache, _user_code_context


class AccessorDataset:
    """Draw from every sanctioned accessor inside dataset code."""

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> tuple[float, float, float]:
        import torch

        torch_value = float(torch.rand((), generator=rng()).item())
        numpy_value = float(rng("numpy").random())
        random_value = rng("random").random()
        return torch_value, numpy_value, random_value


class RngAccessorTest(unittest.TestCase):
    """Exercise stage scope, stream identity, and process-tier wiring."""

    def test_access_outside_user_stage_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "only while user code"):
            rng()

    def test_invalid_kind_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "torch, numpy, or random"):
            rng("ambient")

    def test_generators_are_stable_within_sample_and_independent(self) -> None:
        import numpy as np
        import torch

        sample = _hyperloader._sample_rng_context(17, 3, 11)
        with _user_code_context(sample):
            self.assertIs(rng(), rng())
            self.assertIs(rng("numpy"), rng("numpy"))
            self.assertIs(rng("random"), rng("random"))

            torch_words = _hyperloader._rng_block_from_key(sample[1], 11, 0, 4)
            torch_seed = torch_words[0] | (torch_words[1] << 32)
            expected_torch = torch.rand(
                (), generator=torch.Generator().manual_seed(torch_seed)
            )
            expected_numpy = np.random.Generator(
                np.random.Philox(key=sample[1] ^ _splitmix64(5), counter=11)
            ).random()
            expected_random = PhiloxRandom(stream_id=6)
            expected_random.rekey(sample[1], 11)

            self.assertEqual(
                torch.rand((), generator=rng()).item(), expected_torch.item()
            )
            self.assertEqual(rng("numpy").random(), expected_numpy)
            self.assertEqual(rng("random").random(), expected_random.random())

    def test_process_path_replays_accessors_exactly(self) -> None:
        first_loader = DataLoader(
            AccessorDataset(), batch_size=None, num_workers=1, seed=29
        )
        second_loader = DataLoader(
            AccessorDataset(), batch_size=None, num_workers=1, seed=29
        )
        try:
            self.assertEqual(list(first_loader), list(second_loader))
        finally:
            first_loader.close()
            second_loader.close()

    def test_numpy_accessor_retains_its_state_container_across_samples(self) -> None:
        first = _hyperloader._sample_rng_context(71, 2, 3)
        second = _hyperloader._sample_rng_context(71, 2, 4)
        with _user_code_context(first):
            generator = rng("numpy")
        retained = _cache().numpy_state
        with _user_code_context(second):
            self.assertIs(rng("numpy"), generator)
        self.assertIs(_cache().numpy_state, retained)
        self.assertEqual(
            generator.bit_generator.state["state"]["counter"].tolist(),
            [4, 0, 0, 0],
        )

    def test_nested_and_exceptional_scopes_restore_the_prior_sample(self) -> None:
        outer = _hyperloader._sample_rng_context(73, 2, 5)
        inner = _hyperloader._sample_rng_context(73, 2, 6)
        with _user_code_context(outer):
            outer_generator = rng("random")
            with self.assertRaisesRegex(
                RuntimeError, "sentinel"
            ), _user_code_context(inner):
                self.assertIs(rng("random"), outer_generator)
                raise RuntimeError("sentinel")
            self.assertIs(rng("random"), outer_generator)
        with self.assertRaisesRegex(RuntimeError, "only while user code"):
            rng("random")


if __name__ == "__main__":
    unittest.main()

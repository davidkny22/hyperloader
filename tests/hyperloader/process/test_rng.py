"""Worker-local per-sample RNG installation checks."""

from __future__ import annotations

import random
import unittest

import numpy as np
import torch
from torch.utils.data import get_worker_info
from torch.utils.data._utils import worker as worker_module

from hyperloader import _hyperloader
from hyperloader.process.numpy_surface import _splitmix64
from hyperloader.process.random_surface import PhiloxRandom
from hyperloader.process.rng import WorkerRngContext


class WorkerRngContextTest(unittest.TestCase):
    """Exercise exact seeded globals and retained worker identity."""

    def setUp(self) -> None:
        self._random_state = random.getstate()
        self._numpy_state = np.random.get_state()
        self._torch_state = torch.get_rng_state()

    def tearDown(self) -> None:
        random.setstate(self._random_state)
        np.random.set_state(self._numpy_state)
        torch.set_rng_state(self._torch_state)

    def test_install_matches_each_derived_seed_stream(self) -> None:
        dataset = [0]
        context = WorkerRngContext(2, 4)
        context.attach_dataset(dataset)
        try:
            torch_seed, key = _hyperloader._sample_rng_context(17, 3, 11)
            context.install(17, 3, 11)
            self.assertIsNone(worker_module._worker_info)
            info = get_worker_info()

            random_reference = PhiloxRandom()
            random_reference.rekey(key, 11)
            numpy_reference = np.random.Generator(
                np.random.Philox(key=key ^ _splitmix64(8), counter=11)
            )
            expected_torch = torch.rand(
                (), generator=torch.Generator().manual_seed(torch_seed)
            )

            self.assertEqual(info.id, 2)
            self.assertEqual(info.num_workers, 4)
            self.assertEqual(info.seed, torch_seed)
            self.assertIs(info.dataset, dataset)
            self.assertEqual(random.random(), random_reference.random())
            self.assertEqual(np.random.random(), numpy_reference.random())
            self.assertEqual(torch.rand(()).item(), expected_torch.item())
        finally:
            context.clear()

        self.assertIsNone(get_worker_info())

    def test_install_preserves_each_worker_info_value(self) -> None:
        context = WorkerRngContext(0, 1)
        context.attach_dataset([0])
        try:
            context.install(5, 0, 0)
            first = get_worker_info()
            context.install(5, 0, 1)
            second = get_worker_info()

            self.assertIsNot(first, second)
            self.assertNotEqual(first.seed, second.seed)
        finally:
            context.clear()

    def test_sample_without_worker_info_call_does_not_construct_identity(self) -> None:
        context = WorkerRngContext(0, 1)
        context.attach_dataset([0])
        try:
            context.install(7, 0, 0)

            self.assertIsNone(worker_module._worker_info)
            self.assertIsNotNone(context._worker_info)
            self.assertIsNone(context._worker_info._current)
        finally:
            context.clear()

    def test_reinstall_reproduces_mixed_module_draws(self) -> None:
        context = WorkerRngContext(0, 1)
        context.attach_dataset([0])
        try:
            context.install(19, 2, 7)
            first = (
                random.random(),
                random.getrandbits(71),
                random.randrange(10_000),
                np.random.random(),
                np.random.randint(3, 91),
                np.random.normal(),
            )
            context.install(19, 2, 7)
            second = (
                random.random(),
                random.getrandbits(71),
                random.randrange(10_000),
                np.random.random(),
                np.random.randint(3, 91),
                np.random.normal(),
            )
            self.assertEqual(first, second)
        finally:
            context.clear()

    def test_python_random_uses_the_named_engine_stream(self) -> None:
        context = WorkerRngContext(0, 1)
        context.attach_dataset([0])
        try:
            _, key = _hyperloader._sample_rng_context(41, 7, 23)
            words = _hyperloader._rng_block_from_key(key, 23, 0, 7)
            expected = ((words[0] >> 5) * 67_108_864.0 + (words[1] >> 6)) * (
                2.0**-53
            )

            context.install(41, 7, 23)

            self.assertEqual(random.random(), expected)
            self.assertEqual(
                random.getrandbits(37), words[2] | ((words[3] >> 27) << 32)
            )
        finally:
            context.clear()

    def test_numpy_legacy_state_is_constructed_only_on_use(self) -> None:
        context = WorkerRngContext(0, 1)
        context.attach_dataset([0])
        try:
            context.install(23, 4, 9)
            self.assertIsNone(context._numpy._legacy_state)
            np.random.random()
            np.random.randn()
            self.assertIsNone(context._numpy._legacy_state)

            state = np.random.get_state()

            self.assertIsNotNone(context._numpy._legacy_state)
            self.assertEqual(state[0], "MT19937")
        finally:
            context.clear()

    def test_numpy_rekey_clears_counter_buffer_and_uint32_cache(self) -> None:
        context = WorkerRngContext(0, 1)
        context.attach_dataset([0])
        try:
            _, key = _hyperloader._sample_rng_context(37, 6, 17)
            context._numpy._buffer.fill(91)
            context._numpy._state["buffer_pos"] = 1
            context._numpy._state["has_uint32"] = 1
            context._numpy._state["uinteger"] = 73
            context.install(37, 6, 17)
            state = context._numpy._bit_generator.state

            self.assertEqual(state["state"]["counter"].tolist(), [17, 0, 0, 0])
            self.assertEqual(
                state["state"]["key"].tolist(), [key ^ _splitmix64(8), 0]
            )
            self.assertEqual(state["buffer"].tolist(), [0, 0, 0, 0])
            self.assertEqual(state["buffer_pos"], 4)
            self.assertEqual(state["has_uint32"], 0)
            self.assertEqual(state["uinteger"], 0)
        finally:
            context.clear()

    def test_python_state_round_trips_partial_block_and_gaussian_cache(self) -> None:
        context = WorkerRngContext(0, 1)
        context.attach_dataset([0])
        try:
            context.install(29, 5, 13)
            random.getrandbits(11)
            random.gauss(0.0, 1.0)
            state = random.getstate()
            expected = (random.random(), random.getrandbits(65), random.gauss(0.0, 1.0))

            random.setstate(state)

            self.assertEqual(
                (random.random(), random.getrandbits(65), random.gauss(0.0, 1.0)),
                expected,
            )
        finally:
            context.clear()


if __name__ == "__main__":
    unittest.main()

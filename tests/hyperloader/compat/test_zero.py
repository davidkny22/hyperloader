"""Torch-compatible calling-process execution and continuation."""

from __future__ import annotations

import random
import unittest

import numpy as np
import torch
from hyperloader import DataLoader
from hyperloader.compat.rng import capture_globals
from torch.utils.data import get_worker_info


class AmbientDataset:
    """Expose order, ambient RNG draws, and main-process worker identity."""

    def __init__(self, length: int = 12) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return (
            index,
            random.getrandbits(32),
            int(np.random.randint(0, 1 << 31)),
            int(torch.randint(0, 1 << 31, ()).item()),
            int(get_worker_info() is None),
        )


def _reset_globals(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _records(loader) -> list[tuple[int, ...]]:
    records = []
    for batch in loader:
        records.extend(
            tuple(int(value) for value in row)
            for row in zip(*(column.tolist() for column in batch), strict=True)
        )
    return records


class CompatZeroTest(unittest.TestCase):
    """Match torch draw order and restore ambient calling-process state."""

    def test_default_execution_matches_torch_with_eager_base_seed(self) -> None:
        _reset_globals(31)
        reference = torch.utils.data.DataLoader(
            AmbientDataset(),
            batch_size=3,
            shuffle=True,
            generator=_generator(401),
            num_workers=0,
        )
        reference_records = _records(reference)

        _reset_globals(31)
        candidate = DataLoader(
            AmbientDataset(),
            batch_size=3,
            shuffle=True,
            generator=_generator(401),
            mode="torch-compat",
        )
        try:
            self.assertEqual(_records(candidate), reference_records)
            self.assertEqual(candidate.num_workers, 0)
            self.assertFalse(candidate.persistent_workers)
            self.assertIsNone(candidate.prefetch_factor)
            self.assertTrue(candidate.stats()["enabled"])
        finally:
            candidate.close()

    def test_construction_does_not_consume_the_global_torch_generator(self) -> None:
        _reset_globals(43)
        before = torch.get_rng_state().clone()
        candidate = DataLoader(
            AmbientDataset(),
            batch_size=2,
            shuffle=True,
            mode="torch-compat",
        )
        try:
            self.assertTrue(torch.equal(torch.get_rng_state(), before))
            iter(candidate)
            self.assertFalse(torch.equal(torch.get_rng_state(), before))
        finally:
            candidate.close()

    def test_resume_restores_globals_and_exact_remaining_stream(self) -> None:
        _reset_globals(59)
        reference = torch.utils.data.DataLoader(
            AmbientDataset(),
            batch_size=2,
            shuffle=True,
            generator=_generator(811),
            num_workers=0,
        )
        baseline = _records(reference)

        _reset_globals(59)
        source = DataLoader(
            AmbientDataset(),
            batch_size=2,
            shuffle=True,
            generator=_generator(811),
            mode="torch-compat",
        )
        iterator = iter(source)
        prefix = []
        for _ in range(3):
            batch = next(iterator)
            prefix.extend(
                tuple(int(value) for value in row)
                for row in zip(*(column.tolist() for column in batch), strict=True)
            )
        state = source.state_dict()
        source.close()

        _reset_globals(997)
        resumed = DataLoader(
            AmbientDataset(),
            batch_size=2,
            shuffle=True,
            generator=_generator(811),
            mode="torch-compat",
        )
        try:
            resumed.load_state_dict(state)
            self.assertEqual(capture_globals(), state["current_globals"])
            self.assertEqual(prefix + _records(resumed), baseline)
        finally:
            resumed.close()


if __name__ == "__main__":
    unittest.main()

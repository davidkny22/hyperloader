"""Installed differential for torch-compatible zero-worker execution."""

from __future__ import annotations

import json
import os
import random
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import hyperloader
import numpy as np
import torch
from hyperloader import DataLoader, _hyperloader
from hyperloader.compat import zero
from hyperloader.compat.rng import capture_globals
from torch.utils.data import get_worker_info


class GoldenDataset:
    """Expose torch-visible order and every ambient CPU RNG surface."""

    def __len__(self) -> int:
        return 17

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


def _reference(seed: int | None, shuffle: bool) -> list[tuple[int, ...]]:
    _reset_globals(73)
    return _records(
        torch.utils.data.DataLoader(
            GoldenDataset(),
            batch_size=4,
            shuffle=shuffle,
            generator=None if seed is None else _generator(seed),
            num_workers=0,
        )
    )


def _candidate(seed: int | None, shuffle: bool) -> list[tuple[int, ...]]:
    _reset_globals(73)
    loader = DataLoader(
        GoldenDataset(),
        batch_size=4,
        shuffle=shuffle,
        generator=None if seed is None else _generator(seed),
        mode="torch-compat",
    )
    try:
        return _records(loader)
    finally:
        loader.close()


def _skip_base_seed(loader):
    state = (
        torch.get_rng_state()
        if loader._compat_generator is None
        else loader._compat_generator.get_state()
    )
    iterator = iter(loader._compat_loader)
    if loader._compat_generator is None:
        torch.set_rng_state(state)
    else:
        loader._compat_generator.set_state(state)
    return iterator


class CompatZeroGate(unittest.TestCase):
    """Compare public calling-process streams with pinned torch."""

    def test_pinned_torch_streams_match_bit_exactly(self) -> None:
        expected_root = os.environ.get("HYPERLOADER_EXPECTED_INSTALL_ROOT")
        if expected_root is not None:
            root = Path(expected_root).resolve()
            self.assertTrue(Path(hyperloader.__file__).resolve().is_relative_to(root))
            self.assertTrue(Path(_hyperloader.__file__).resolve().is_relative_to(root))

        mutation = (
            mock.patch.object(zero, "_start_torch_iterator", _skip_base_seed)
            if os.environ.get("HYPERLOADER_COMPAT_ZERO_MUTATION") == "skip-base-seed"
            else nullcontext()
        )
        evidence = []
        with mutation:
            for seed, shuffle in (
                (401, False),
                (401, True),
                (977, True),
                (None, True),
            ):
                with self.subTest(seed=seed, shuffle=shuffle):
                    reference = _reference(seed, shuffle)
                    candidate = _candidate(seed, shuffle)
                    self.assertEqual(candidate, reference)
                    evidence.append(
                        {
                            "records": len(candidate),
                            "seed": seed,
                            "shuffle": shuffle,
                        }
                    )

        self.assertNotEqual(_candidate(401, True), _candidate(402, True))
        metrics_path = os.environ.get("HYPERLOADER_COMPAT_ZERO_METRICS")
        if metrics_path is not None:
            Path(metrics_path).write_text(
                json.dumps(
                    {"cases": evidence, "torch": torch.__version__},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

    def test_checkpoint_restores_ambient_globals_and_remaining_stream(self) -> None:
        baseline = _reference(811, True)
        _reset_globals(73)
        source = DataLoader(
            GoldenDataset(),
            batch_size=4,
            shuffle=True,
            generator=_generator(811),
            mode="torch-compat",
        )
        iterator = iter(source)
        prefix = []
        for _ in range(2):
            batch = next(iterator)
            prefix.extend(
                tuple(int(value) for value in row)
                for row in zip(*(column.tolist() for column in batch), strict=True)
            )
        state = source.state_dict()
        source.close()

        _reset_globals(1237)
        resumed = DataLoader(
            GoldenDataset(),
            batch_size=4,
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

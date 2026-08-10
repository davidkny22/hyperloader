"""Installed public gate for result-invariant controller adaptation."""

from __future__ import annotations

import os
import random
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from hyperloader import DataLoader, HyperConfig, _hyperloader
from hyperloader.config import ControlConfig, FactorConfig
from hyperloader.control import AdaptiveController
from hyperloader.process.frontier import FrontierRuntime


class SeededDataset:
    """Return result-observable values from every provided global RNG surface."""

    def __init__(self, length: int = 24) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, int | float]:
        return {
            "index": index,
            "numpy": float(np.random.random()),
            "python": random.random(),
            "torch": torch.rand(()).item(),
        }


def _stream_and_widths(*, stalled: bool, seed: int) -> tuple[list[object], list[int]]:
    config = HyperConfig(
        control=ControlConfig(cadence=1e-9),
        factors=FactorConfig(hysteresis=1),
    )
    loader = DataLoader(
        SeededDataset(),
        batch_size=1,
        num_workers=4,
        seed=seed,
        config=config,
    )
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                FrontierRuntime, "consume_stall_flag", return_value=stalled
            )
        )
        if os.environ.get("HYPERLOADER_ADAPTATION_MUTATION") == "freeze-controller":
            stack.enter_context(
                mock.patch.object(AdaptiveController, "observe", return_value=None)
            )
        try:
            stream = [
                (
                    int(batch["index"].item()),
                    float(batch["numpy"].item()),
                    float(batch["python"].item()),
                    float(batch["torch"].item()),
                )
                for batch in loader
            ]
            widths = [decision.width for decision in loader._controller.decisions]
        finally:
            loader.close()
    return stream, widths


class AdaptationInvarianceGate(unittest.TestCase):
    """Force divergent worker schedules without changing the seeded stream."""

    def test_forced_divergent_schedules_preserve_the_stream(self) -> None:
        expected_root = os.environ.get("HYPERLOADER_EXPECTED_INSTALL_ROOT")
        if expected_root is not None:
            self.assertTrue(
                Path(_hyperloader.__file__)
                .resolve()
                .is_relative_to(Path(expected_root).resolve())
            )

        shrinking, shrinking_widths = _stream_and_widths(stalled=False, seed=313)
        fed, fed_widths = _stream_and_widths(stalled=True, seed=313)

        self.assertEqual(shrinking, fed)
        self.assertNotEqual(shrinking_widths, fed_widths)
        self.assertIn(1, shrinking_widths)
        self.assertTrue(fed_widths)
        self.assertEqual(set(fed_widths), {4})

    def test_root_seed_is_a_contract_input(self) -> None:
        first, _ = _stream_and_widths(stalled=False, seed=313)
        changed, _ = _stream_and_widths(stalled=False, seed=314)

        self.assertNotEqual(first, changed)


if __name__ == "__main__":
    unittest.main()

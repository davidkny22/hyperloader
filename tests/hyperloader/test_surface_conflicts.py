"""Differential constructor-conflict checks against an installed torch reference."""

from __future__ import annotations

import unittest
from collections.abc import Callable
from typing import Any

from hyperloader import DataLoader


def _outcome(loader: Callable[..., Any], arguments: dict[str, Any]) -> str:
    try:
        loader([], **arguments)
    except (TypeError, ValueError):
        return "rejected"
    return "accepted"


class SurfaceConflictTest(unittest.TestCase):
    """Require matching acceptance for invariant torch constructor rules."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest("torch is unavailable for conflict comparison") from error
        cls.reference = torch.utils.data.DataLoader

    def test_invariant_conflict_matrix_matches_torch(self) -> None:
        cases = {
            "sampler with shuffle": {"shuffle": True, "sampler": [0]},
            "batch sampler conflict": {"batch_size": 2, "batch_sampler": [[0]]},
            "batch sampler default": {"batch_sampler": [[0]]},
            "negative workers": {"num_workers": -1},
            "negative timeout": {"timeout": -1},
            "zero-worker prefetch": {"num_workers": 0, "prefetch_factor": 2},
        }

        for name, arguments in cases.items():
            with self.subTest(name=name):
                self.assertEqual(
                    _outcome(DataLoader, arguments),
                    _outcome(self.reference, arguments),
                )


if __name__ == "__main__":
    unittest.main()

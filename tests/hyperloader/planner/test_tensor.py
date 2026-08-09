"""Registered contiguous-tensor planner checks."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from hyperloader import _hyperloader
from hyperloader.planner import BlackBoxPlan, TensorPlan, build_plan


class TensorPlanTest(unittest.TestCase):
    """Verify registration selection, sampler parity, and refuge behavior."""

    def test_exact_torch_tensor_selects_registered_plan(self) -> None:
        plan = build_plan(torch.arange(12).reshape(4, 3), False)

        self.assertIsInstance(plan, TensorPlan)
        self.assertEqual(plan.length, 4)

    def test_shuffle_uses_native_contract_permutation(self) -> None:
        plan = build_plan(torch.arange(24).reshape(8, 3), True)
        actual = [plan.index(11, 3, position) for position in range(8)]
        expected = [
            _hyperloader._permutation_index(11, 3, 8, position) for position in range(8)
        ]

        self.assertEqual(actual, expected)

    def test_missing_registration_returns_to_black_box_refuge(self) -> None:
        with mock.patch("hyperloader.planner.registry._load_mappings", return_value=()):
            plan = build_plan(torch.arange(4), False)

        self.assertIsInstance(plan, BlackBoxPlan)


if __name__ == "__main__":
    unittest.main()

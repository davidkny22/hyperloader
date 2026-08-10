"""Data-driven planner registry validation and refuge tests."""

from __future__ import annotations

import unittest
from unittest import mock

from hyperloader.planner import BlackBoxPlan, build_plan
from hyperloader.planner.registry import _load_mappings


class RegistryTest(unittest.TestCase):
    """Prove mapping data is complete, unique, and safely resolved."""

    def test_launch_mapping_ids_and_builders_are_unique(self) -> None:
        rows = _load_mappings()
        ids = [row["id"] for row in rows]
        builders = [row["builder"] for row in rows]

        self.assertEqual(len(rows), 6)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(builders), len(set(builders)))
        self.assertEqual({row["match"] for row in rows}, {"exact", "subclass"})

    def test_unresolvable_builder_returns_to_black_box(self) -> None:
        row = {
            "id": "missing-builder",
            "dataset_module": "builtins",
            "dataset_type": "list",
            "match": "exact",
            "builder": "hyperloader.planner.missing:build_plan",
        }
        with mock.patch(
            "hyperloader.planner.registry._load_mappings", return_value=(row,)
        ):
            plan = build_plan([1, 2], False)

        self.assertIsInstance(plan, BlackBoxPlan)

    def test_unrecognized_dataset_keeps_black_box_refuge(self) -> None:
        plan = build_plan(range(3), False)

        self.assertIsInstance(plan, BlackBoxPlan)


if __name__ == "__main__":
    unittest.main()

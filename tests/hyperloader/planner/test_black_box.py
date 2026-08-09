"""Black-box map planning and sampler index tests."""

import unittest

from hyperloader import _hyperloader
from hyperloader.planner import build_black_box_plan


class UnsizedDataset:
    """Represent an iterable plan that has no map-style length."""

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter((1, 2, 3))


class BlackBoxPlanTest(unittest.TestCase):
    """Exercise refuge selection and native sampler coordinates."""

    def test_sized_dataset_selects_identity_black_box_plan(self) -> None:
        plan = build_black_box_plan(list(range(4)), False)

        self.assertIsNotNone(plan)
        self.assertEqual([plan.index(7, 0, position) for position in range(4)], [0, 1, 2, 3])

    def test_shuffle_uses_native_contract_permutation(self) -> None:
        plan = build_black_box_plan(list(range(8)), True)
        actual = [plan.index(11, 3, position) for position in range(8)]
        expected = [
            _hyperloader._permutation_index(11, 3, 8, position)
            for position in range(8)
        ]

        self.assertEqual(actual, expected)

    def test_unsized_dataset_defers_to_iterable_planning(self) -> None:
        self.assertIsNone(build_black_box_plan(UnsizedDataset(), False))


if __name__ == "__main__":
    unittest.main()

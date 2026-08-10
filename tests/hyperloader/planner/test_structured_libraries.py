"""Torchvision and Hugging Face structure-plan tests."""

from __future__ import annotations

import unittest

from datasets import Dataset
from torchvision.datasets.folder import DatasetFolder

from hyperloader import DataLoader
from hyperloader.planner import StructurePlan, build_plan


def decode_path(path: str) -> int:
    """Decode an in-memory path token."""
    return int(path)


def double(value: int) -> int:
    """Apply a stable sample transform."""
    return value * 2


def increment(value: int) -> int:
    """Apply a stable target transform."""
    return value + 1


class MemoryFolder(DatasetFolder):
    """Expose DatasetFolder's canonical fields without filesystem discovery."""

    def __init__(self) -> None:
        self.samples = [("1", 4), ("2", 5), ("3", 6)]
        self.loader = decode_path
        self.transform = double
        self.target_transform = increment


class StructuredLibraryPlanTest(unittest.TestCase):
    """Exercise family matching, adapter construction, and public execution."""

    def test_torchvision_subclass_decomposes_canonical_operations(self) -> None:
        dataset = MemoryFolder()
        plan = build_plan(dataset, False)

        self.assertIsInstance(plan, StructurePlan)
        self.assertEqual(plan.mapping_id, "torchvision-dataset-folder")
        self.assertEqual(
            [stage.name for stage in plan.stages],
            ["sample-path", "loader-decode", "sample-transform", "target-transform"],
        )
        self.assertEqual(plan.execution_dataset[1], dataset[1])

    def test_torchvision_plan_runs_through_public_process_path(self) -> None:
        loader = DataLoader(MemoryFolder(), batch_size=None, num_workers=1, seed=83)
        try:
            self.assertEqual(list(loader), [(2, 5), (4, 6), (6, 7)])
        finally:
            loader.close()

    def test_huggingface_format_state_is_preserved(self) -> None:
        dataset = Dataset.from_dict({"x": [1, 2], "y": [3.5, 4.5]}).with_format("numpy")
        plan = build_plan(dataset, False)

        self.assertIsInstance(plan, StructurePlan)
        self.assertEqual(plan.mapping_id, "huggingface-arrow-dataset")
        for index in range(len(dataset)):
            actual = plan.execution_dataset[index]
            expected = dataset[index]
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self.assertEqual(actual[key], expected[key])


if __name__ == "__main__":
    unittest.main()

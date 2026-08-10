"""Tensor, memory-map, and Parquet structure-plan tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as arrow_dataset
import pyarrow.parquet as parquet
import torch
from torch.utils.data import TensorDataset

from hyperloader import DataLoader
from hyperloader.planner import StructurePlan, build_plan


class StructuredArrayPlanTest(unittest.TestCase):
    """Exercise row-addressable launch mappings and their process adapters."""

    def test_tensor_dataset_decomposes_to_row_views(self) -> None:
        dataset = TensorDataset(torch.arange(6).reshape(3, 2), torch.arange(3))
        plan = build_plan(dataset, False)

        self.assertIsInstance(plan, StructurePlan)
        self.assertEqual(plan.mapping_id, "torch-tensor-dataset")
        for actual, expected in zip(plan.execution_dataset[1], dataset[1], strict=True):
            self.assertTrue(torch.equal(actual, expected))

    def test_memmap_reopens_and_delivers_rows_through_public_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.bin"
            dataset = np.memmap(path, dtype=np.int32, mode="w+", shape=(4, 2))
            dataset[:] = np.arange(8, dtype=np.int32).reshape(4, 2)
            dataset.flush()
            plan = build_plan(dataset, False)

            self.assertIsInstance(plan, StructurePlan)
            self.assertEqual(plan.mapping_id, "numpy-memmap")

            loader = None
            try:
                loader = DataLoader(dataset, batch_size=2, num_workers=1, seed=89)
                self.assertEqual(
                    [batch.tolist() for batch in loader],
                    [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
                )
                fingerprint = {
                    element.path: element.value
                    for element in loader._fingerprint.elements
                }
                self.assertEqual(
                    fingerprint["batch_shape"],
                    {
                        "dtype": "torch.int32",
                        "kind": "tensor",
                        "shape": [2, 2],
                        "source": "probe",
                    },
                )
            finally:
                if loader is not None:
                    loader.close()
                dataset._mmap.close()

    def test_local_parquet_dataset_maps_rows_without_partition_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.parquet"
            parquet.write_table(pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]}), path)
            dataset = arrow_dataset.dataset(path, format="parquet")
            plan = build_plan(dataset, False)

            self.assertIsInstance(plan, StructurePlan)
            self.assertEqual(plan.mapping_id, "pyarrow-parquet-dataset")
            self.assertEqual(plan.execution_dataset[1], {"x": 2, "y": "b"})

            loader = DataLoader(dataset, batch_size=None, num_workers=1, seed=97)
            try:
                self.assertEqual(
                    list(loader),
                    [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}, {"x": 3, "y": "c"}],
                )
            finally:
                loader.close()


if __name__ == "__main__":
    unittest.main()

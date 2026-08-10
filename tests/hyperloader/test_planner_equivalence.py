"""Installed public gate for mapped-plan and black-box equivalence."""

from __future__ import annotations

import inspect
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.dataset as arrow_dataset
import pyarrow.parquet as parquet
import torch
from datasets import Dataset
from torch.utils.data import TensorDataset

from hyperloader import DataLoader, _hyperloader
from hyperloader.planner import BlackBoxPlan, StructurePlan

from .planner_equivalence_support import (
    DelegatingDataset,
    IncompleteFolder,
    MemoryFolder,
    ParquetRows,
    assert_contract_equal,
    shifted_structure_index,
)


class PlannerEquivalenceGate(unittest.TestCase):
    """Compare every launch mapping with its black-box public execution."""

    def test_installed_product_is_under_the_declared_root(self) -> None:
        expected_root = os.environ.get("HYPERLOADER_EXPECTED_INSTALL_ROOT")
        if expected_root is None:
            self.skipTest("the installed-artifact root is declared by the gate harness")
        root = Path(expected_root).resolve()
        self.assertTrue(Path(_hyperloader.__file__).resolve().is_relative_to(root))
        self.assertTrue(
            Path(inspect.getfile(StructurePlan)).resolve().is_relative_to(root)
        )

    def test_contiguous_tensor_matches_black_box(self) -> None:
        dataset = torch.arange(24, dtype=torch.int64).reshape(12, 2)
        self._assert_stream_equal(dataset, DelegatingDataset(dataset), seed=101)

    def test_tensor_dataset_matches_black_box(self) -> None:
        dataset = TensorDataset(
            torch.arange(24, dtype=torch.int64).reshape(12, 2),
            torch.linspace(0.0, 1.0, 12, dtype=torch.float64),
        )
        self._assert_stream_equal(dataset, DelegatingDataset(dataset), seed=103)

    def test_torchvision_folder_matches_black_box(self) -> None:
        dataset = MemoryFolder(length=11)
        self._assert_stream_equal(dataset, DelegatingDataset(dataset), seed=107)

    def test_huggingface_arrow_matches_black_box(self) -> None:
        dataset = Dataset.from_dict(
            {"token": list(range(11)), "score": [index / 7 for index in range(11)]}
        ).with_format("numpy")
        self._assert_stream_equal(dataset, DelegatingDataset(dataset), seed=109)

    def test_numpy_memmap_matches_black_box(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.bin"
            dataset = np.memmap(path, dtype=np.int32, mode="w+", shape=(11, 3))
            dataset[:] = np.arange(33, dtype=np.int32).reshape(11, 3)
            dataset.flush()
            try:
                self._assert_stream_equal(dataset, DelegatingDataset(dataset), seed=113)
            finally:
                dataset._mmap.close()

    def test_local_parquet_matches_black_box(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.parquet"
            parquet.write_table(
                pa.table(
                    {
                        "token": list(range(11)),
                        "label": [f"row-{index}" for index in range(11)],
                    }
                ),
                path,
            )
            dataset = arrow_dataset.dataset(path, format="parquet")
            self._assert_stream_equal(dataset, ParquetRows(dataset), seed=127)

    def test_exception_delivery_position_matches_black_box(self) -> None:
        dataset = MemoryFolder(exploding=True, length=7)
        mapped = DataLoader(dataset, batch_size=None, num_workers=2, seed=131)
        black_box = DataLoader(
            DelegatingDataset(dataset), batch_size=None, num_workers=2, seed=131
        )
        try:
            expected_values, expected_error = self._consume_until_error(black_box)
            actual_values, actual_error = self._consume_until_error(mapped)
            assert_contract_equal(self, actual_values, expected_values)
            self.assertEqual(actual_values, [(0, 11), (2, 12), (4, 13)])
            self.assertIs(type(actual_error), type(expected_error))
            for error in (actual_error, expected_error):
                self.assertIn(
                    "Caught ValueError in DataLoader worker process", str(error)
                )
                self.assertIn("planner equivalence sentinel", str(error))
        finally:
            mapped.close()
            black_box.close()

    def test_changed_seed_changes_the_mapped_stream(self) -> None:
        first = DataLoader(
            MemoryFolder(length=11),
            batch_size=None,
            shuffle=True,
            num_workers=2,
            seed=137,
        )
        changed = DataLoader(
            MemoryFolder(length=11),
            batch_size=None,
            shuffle=True,
            num_workers=2,
            seed=139,
        )
        try:
            with self.assertRaises(AssertionError):
                assert_contract_equal(self, list(first), list(changed))
        finally:
            first.close()
            changed.close()

    def test_incomplete_registered_family_uses_black_box_refuge(self) -> None:
        loader = DataLoader(
            IncompleteFolder(), batch_size=None, num_workers=1, seed=149
        )
        try:
            self.assertIsInstance(loader._plan, BlackBoxPlan)
            self.assertEqual(list(loader), [("refuge", 0), ("refuge", 1)])
        finally:
            loader.close()

    def _assert_stream_equal(
        self, mapped_dataset: Any, black_box_dataset: Any, *, seed: int
    ) -> None:
        black_box = DataLoader(
            black_box_dataset,
            batch_size=None,
            shuffle=True,
            num_workers=2,
            seed=seed,
        )
        mutation = (
            mock.patch.object(StructurePlan, "index", shifted_structure_index)
            if os.environ.get("HYPERLOADER_PLANNER_MUTATION") == "rotate-position"
            else nullcontext()
        )
        mapped = None
        try:
            expected = list(black_box)
            with mutation:
                mapped = DataLoader(
                    mapped_dataset,
                    batch_size=None,
                    shuffle=True,
                    num_workers=2,
                    seed=seed,
                )
                actual = list(mapped)
            assert_contract_equal(self, actual, expected)
        finally:
            if mapped is not None:
                mapped.close()
            black_box.close()

    @staticmethod
    def _consume_until_error(loader: DataLoader) -> tuple[list[Any], BaseException]:
        values = []
        iterator = iter(loader)
        while True:
            try:
                values.append(next(iterator))
            except BaseException as error:
                if isinstance(error, StopIteration):
                    raise AssertionError("the expected dataset exception did not occur")
                return values, error


if __name__ == "__main__":
    unittest.main()

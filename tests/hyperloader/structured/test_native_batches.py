"""Public-path assurance for batch-native structured sources."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as arrow_dataset
import pyarrow.parquet as parquet
import torch
from datasets import Dataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset, default_collate

from hyperloader import DataLoader


def assert_tree_equal(case: unittest.TestCase, actual: Any, expected: Any) -> None:
    """Compare nested batch values, tensor metadata, and bits."""
    if isinstance(expected, torch.Tensor):
        case.assertIsInstance(actual, torch.Tensor)
        case.assertEqual(actual.dtype, expected.dtype)
        case.assertEqual(actual.shape, expected.shape)
        case.assertEqual(actual.stride(), expected.stride())
        case.assertTrue(torch.equal(actual, expected))
        return
    if isinstance(expected, dict):
        case.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            assert_tree_equal(case, actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        case.assertIsInstance(actual, type(expected))
        case.assertEqual(len(actual), len(expected))
        for actual_item, expected_item in zip(actual, expected, strict=True):
            assert_tree_equal(case, actual_item, expected_item)
        return
    case.assertEqual(actual, expected)


class NativeBatchPathTest(unittest.TestCase):
    """Prove storage ownership and batched conversion through DataLoader."""

    def test_tensor_dataset_batches_are_source_storage_views(self) -> None:
        features = torch.arange(24, dtype=torch.int64).reshape(6, 4)
        labels = torch.arange(6, dtype=torch.float32)
        dataset = TensorDataset(features, labels)
        expected = next(iter(TorchDataLoader(dataset, batch_size=3)))
        loader = DataLoader(dataset, batch_size=3, num_workers=2, seed=11)
        try:
            iterator = iter(loader)
            actual = next(iterator)
            assert_tree_equal(self, actual, expected)
            self.assertIsNone(loader._process_pool)
            self.assertEqual(
                actual[0].untyped_storage().data_ptr(),
                features.untyped_storage().data_ptr(),
            )
            self.assertEqual(
                actual[1].untyped_storage().data_ptr(),
                labels.untyped_storage().data_ptr(),
            )
            current = loader.stats()["current"]
            self.assertEqual(current["delivered_samples"], 3)
            self.assertEqual(current["delivered_batches"], 1)
            self.assertEqual(
                current["delivered_bytes"],
                actual[0].numel() * actual[0].element_size()
                + actual[1].numel() * actual[1].element_size(),
            )
        finally:
            loader.close()

    def test_memmap_batch_is_a_live_mapped_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            source = np.memmap(path, dtype=np.int32, mode="w+", shape=(6, 3))
            source[:] = np.arange(18, dtype=np.int32).reshape(6, 3)
            source.flush()
            loader = DataLoader(source, batch_size=3, num_workers=2, seed=13)
            batch = next(iter(loader))
            try:
                mapped = loader._execution_dataset._mapped
                self.assertIsNone(loader._process_pool)
                self.assertEqual(batch.data_ptr(), mapped.ctypes.data)
                mapped[0, 0] = 901
                self.assertEqual(batch[0, 0].item(), 901)
                fingerprint = {
                    element.path: element.value
                    for element in loader._fingerprint.elements
                }
                self.assertEqual(fingerprint["batch_shape"]["source"], "probe")
                loader.close()
                self.assertEqual(batch[0, 0].item(), 901)
            finally:
                del batch
                del mapped
                loader.close()
                source._mmap.close()

    def test_arrow_batch_matches_torch_without_row_reconstruction(self) -> None:
        dataset = Dataset.from_dict(
            {
                "token": [1, 2, 3, 4],
                "score": [1.5, 2.5, 3.5, 4.5],
                "text": ["a", "b", "c", "d"],
            }
        )
        expected = next(iter(TorchDataLoader(dataset, batch_size=3)))
        loader = DataLoader(dataset, batch_size=3, num_workers=2, seed=17)
        try:
            actual = next(iter(loader))
            assert_tree_equal(self, actual, expected)
            self.assertIsNone(loader._process_pool)
        finally:
            loader.close()

    def test_parquet_decodes_once_per_delivered_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.parquet"
            parquet.write_table(
                pa.table({"token": [1, 2, 3, 4], "text": ["a", "b", "c", "d"]}),
                path,
            )
            dataset = arrow_dataset.dataset(path, format="parquet")
            loader = DataLoader(dataset, batch_size=2, num_workers=2, seed=19)
            adapter = loader._execution_dataset

            class CountingSource:
                def __init__(self, source: Any) -> None:
                    self.source = source
                    self.calls = 0

                def take(self, indices: list[int]) -> Any:
                    self.calls += 1
                    return self.source.take(indices)

            counter = CountingSource(adapter._dataset)
            adapter._dataset = counter
            expected = default_collate([adapter[0], adapter[1]])
            iterator = iter(loader)
            actual = next(iterator)
            try:
                assert_tree_equal(self, actual, expected)
                self.assertIsNone(loader._process_pool)
                self.assertEqual(counter.calls, 2)
                next(iterator)
                self.assertEqual(counter.calls, 3)
            finally:
                loader.close()


if __name__ == "__main__":
    unittest.main()

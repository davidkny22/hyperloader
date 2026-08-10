"""Per-family dataset fingerprint tests."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as arrow_dataset
import pyarrow.parquet as parquet
import torch
from datasets import Dataset
from torch.utils.data import TensorDataset
from torchvision.datasets.folder import DatasetFolder

from hyperloader.fingerprint import build_dataset_fingerprint
from hyperloader.stages import Collate, Source, Transform, pipeline


class FolderFixture(DatasetFolder):
    """Expose ordered file-target pairs without scanning the root."""

    def __init__(self, root: Path, names: tuple[str, ...]) -> None:
        self.root = str(root)
        self.samples = [(str(root / name), index) for index, name in enumerate(names)]
        self.loader = bytes
        self.transform = None
        self.target_transform = None


def add_one(value: int) -> int:
    """Provide a stable pipeline transform identity."""
    return value + 1


def add_two(value: int) -> int:
    """Provide a distinct pipeline transform identity."""
    return value + 2


def collect(values: list[int]) -> list[int]:
    """Provide a stable pipeline collation identity."""
    return values


class DatasetFingerprintTest(unittest.TestCase):
    """Prove content and strict behavior for every launch storage family."""

    def test_file_inventory_is_root_relative_ordered_and_mtime_free(self) -> None:
        first = tempfile.TemporaryDirectory()
        second = tempfile.TemporaryDirectory()
        self.addCleanup(first.cleanup)
        self.addCleanup(second.cleanup)
        for directory in (Path(first.name), Path(second.name)):
            (directory / "a.bin").write_bytes(b"same")
            (directory / "b.bin").write_bytes(b"size")
        left = FolderFixture(Path(first.name), ("a.bin", "b.bin"))
        right = FolderFixture(Path(second.name), ("a.bin", "b.bin"))

        baseline = build_dataset_fingerprint(left, "content")
        os.utime(Path(first.name) / "a.bin", (1, 1))

        self.assertEqual(build_dataset_fingerprint(left, "content"), baseline)
        self.assertEqual(build_dataset_fingerprint(right, "content"), baseline)
        self.assertNotEqual(
            build_dataset_fingerprint(
                FolderFixture(Path(second.name), ("b.bin", "a.bin")), "content"
            ).digest,
            baseline.digest,
        )

    def test_strict_files_detect_same_size_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sample.bin"
            path.write_bytes(b"left")
            dataset = FolderFixture(root, ("sample.bin",))
            content = build_dataset_fingerprint(dataset, "content")
            strict = build_dataset_fingerprint(dataset, "strict")

            path.write_bytes(b"rght")

            self.assertEqual(build_dataset_fingerprint(dataset, "content"), content)
            self.assertNotEqual(
                build_dataset_fingerprint(dataset, "strict").digest, strict.digest
            )

    def test_tensor_and_memmap_capture_shape_dtype_and_strict_bits(self) -> None:
        tensor = torch.arange(6, dtype=torch.int32).reshape(3, 2)
        baseline = build_dataset_fingerprint(tensor, "content")
        changed_bits = tensor.clone()
        changed_bits[0, 0] = 99

        self.assertEqual(
            build_dataset_fingerprint(changed_bits, "content").digest,
            baseline.digest,
        )
        self.assertNotEqual(
            build_dataset_fingerprint(changed_bits, "strict").digest,
            build_dataset_fingerprint(tensor, "strict").digest,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.bin"
            mapped = np.memmap(path, dtype=np.int16, mode="w+", shape=(3, 2))
            mapped[:] = np.arange(6, dtype=np.int16).reshape(3, 2)
            mapped.flush()
            try:
                values = {
                    element.path: element.value
                    for element in build_dataset_fingerprint(mapped, "content").elements
                }
                self.assertEqual(values["dataset.shape"], [3, 2])
                self.assertEqual(values["dataset.dtype"], "int16")
                self.assertEqual(values["dataset.files[0].path"], "rows.bin")
            finally:
                mapped._mmap.close()

    def test_tensor_dataset_and_pipeline_capture_each_composed_input(self) -> None:
        tensors = TensorDataset(
            torch.arange(6, dtype=torch.int32).reshape(3, 2),
            torch.arange(3, dtype=torch.float32),
        )
        tensor_values = {
            element.path: element.value
            for element in build_dataset_fingerprint(tensors, "content").elements
        }
        self.assertEqual(tensor_values["dataset.tensor_count"], 2)
        self.assertEqual(tensor_values["dataset.tensors[0].shape"], [3, 2])
        self.assertEqual(tensor_values["dataset.tensors[1].dtype"], "torch.float32")

        left = pipeline(
            Source(range(4), output_type=int),
            Transform(add_one, input_type=int, output_type=int),
            Collate(collect, input_type=int, output_type=list),
        )
        right = pipeline(
            Source(range(4), output_type=int),
            Transform(add_two, input_type=int, output_type=int),
            Collate(collect, input_type=int, output_type=list),
        )
        left_fingerprint = build_dataset_fingerprint(left, "content")
        left_values = {
            element.path: element.value for element in left_fingerprint.elements
        }

        self.assertEqual(left_values["dataset.stage_count"], 3)
        self.assertEqual(left_values["dataset.stages[1].input_type"], "builtins.int")
        self.assertNotEqual(
            left_fingerprint.digest,
            build_dataset_fingerprint(right, "content").digest,
        )

    def test_arrow_families_capture_schema_rows_and_parquet_files(self) -> None:
        arrow = Dataset.from_dict({"x": [1, 2], "label": ["a", "b"]})
        arrow_values = {
            element.path: element.value
            for element in build_dataset_fingerprint(arrow, "content").elements
        }
        self.assertEqual(arrow_values["dataset.row_count"], 2)
        self.assertIn("x", arrow_values["dataset.schema"])
        formatted = arrow.with_format("numpy")
        self.assertNotEqual(
            build_dataset_fingerprint(formatted, "content").digest,
            build_dataset_fingerprint(arrow, "content").digest,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.parquet"
            parquet.write_table(pa.table({"x": [1, 2, 3]}), path)
            dataset = arrow_dataset.dataset(path, format="parquet")
            values = {
                element.path: element.value
                for element in build_dataset_fingerprint(dataset, "content").elements
            }
            self.assertEqual(values["dataset.row_count"], 3)
            self.assertEqual(values["dataset.files[0].path"], "rows.parquet")
            self.assertEqual(values["dataset.files[0].size"], path.stat().st_size)


if __name__ == "__main__":
    unittest.main()

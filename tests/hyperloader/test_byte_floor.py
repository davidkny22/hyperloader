"""Installed public gate for per-class loader byte traffic."""

from __future__ import annotations

import inspect
import json
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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset
from torchvision.io import encode_png

from hyperloader import Collate, DataLoader, Decode, Source, _hyperloader, pipeline


def forbidden_decoder(_value: torch.Tensor) -> torch.Tensor:
    """Fail when a selected decoder does not replace the refuge callable."""
    raise AssertionError("the selected decoder was not installed")


class ByteFloorGate(unittest.TestCase):
    """Prove every reachable native class stays at its irreducible byte floor."""

    def test_installed_product_is_under_the_declared_root(self) -> None:
        expected_root = os.environ.get("HYPERLOADER_EXPECTED_INSTALL_ROOT")
        if expected_root is None:
            self.skipTest("the installed-artifact root is declared by the gate harness")
        root = Path(expected_root).resolve()
        self.assertTrue(Path(_hyperloader.__file__).resolve().is_relative_to(root))
        self.assertTrue(
            Path(inspect.getfile(DataLoader)).resolve().is_relative_to(root)
        )

    def test_reachable_fast_path_classes_meet_the_byte_budget(self) -> None:
        reports: dict[str, dict[str, object]] = {}
        self._record_tensor_view(reports)
        self._record_tensor_dataset_view(reports)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._record_memmap_view(root, reports)
            self._record_parquet_decode(root, reports)
        self._record_arrow_transform(reports)
        self._record_pinned_image(reports)
        self._record_variable_text(reports)

        self.assertEqual(
            set(reports),
            {
                "contiguous-tensor",
                "tensor-dataset",
                "numpy-memmap",
                "huggingface-arrow",
                "pyarrow-parquet",
                "pinned-decode",
                "tokenized-text",
            },
        )
        mutation = os.environ.get("HYPERLOADER_BYTE_FLOOR_MUTATION")
        for source_class, report in reports.items():
            with self.subTest(source_class=source_class):
                samples = int(report["produced_samples"])
                actual = float(report["actual_bytes_per_sample"])
                irreducible = float(report["irreducible_bytes_per_sample"])
                if (
                    mutation == "extra-payload-copy"
                    and source_class == "contiguous-tensor"
                ):
                    actual += int(report["payload_bytes"]) / samples
                self.assertLessEqual(
                    actual - irreducible,
                    max(64.0, irreducible * 0.01),
                )
                self.assertEqual(
                    report["actual_bytes"],
                    report["pinned_stage_bytes"] + report["arena_write_bytes"],
                )
        report_path = os.environ.get("HYPERLOADER_BYTE_FLOOR_REPORT")
        if report_path is not None:
            Path(report_path).write_text(
                json.dumps(reports, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def test_shuffled_tensor_is_not_declared_as_a_view_class(self) -> None:
        dataset = torch.arange(512, dtype=torch.float64).reshape(16, 32)
        loader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=1,
            shuffle=True,
            seed=271,
        )
        try:
            batch = next(iter(loader))
            self.assertNotEqual(
                batch.untyped_storage().data_ptr(),
                dataset.untyped_storage().data_ptr(),
            )
            self.assertNotIn("memory", loader.stats())
        finally:
            loader.close()

    def _record_tensor_view(self, reports: dict[str, dict[str, object]]) -> None:
        dataset = torch.arange(512, dtype=torch.float64).reshape(16, 32)
        loader = DataLoader(dataset, batch_size=4, num_workers=1, seed=263)
        try:
            batch = next(iter(loader))
            self.assertEqual(
                batch.untyped_storage().data_ptr(),
                dataset.untyped_storage().data_ptr(),
            )
            self._capture(loader, reports, expected_delivery="view")
        finally:
            loader.close()

    def _record_tensor_dataset_view(
        self, reports: dict[str, dict[str, object]]
    ) -> None:
        dataset = TensorDataset(
            torch.arange(64, dtype=torch.int64).reshape(16, 4),
            torch.arange(16, dtype=torch.int64),
        )
        loader = DataLoader(dataset, batch_size=4, num_workers=1, seed=269)
        try:
            batch = next(iter(loader))
            for value, source in zip(batch, dataset.tensors, strict=True):
                self.assertEqual(
                    value.untyped_storage().data_ptr(),
                    source.untyped_storage().data_ptr(),
                )
            self._capture(loader, reports, expected_delivery="view")
        finally:
            loader.close()

    def _record_memmap_view(
        self, root: Path, reports: dict[str, dict[str, object]]
    ) -> None:
        path = root / "rows.bin"
        dataset = np.memmap(path, dtype=np.float64, mode="w+", shape=(16, 32))
        dataset[:] = np.arange(512, dtype=np.float64).reshape(16, 32)
        dataset.flush()
        loader = DataLoader(dataset, batch_size=4, num_workers=1, seed=277)
        try:
            batch = next(iter(loader))
            self.assertTrue(
                np.shares_memory(batch.numpy(), loader._execution_dataset._array())
            )
            self._capture(loader, reports, expected_delivery="view")
        finally:
            loader.close()
            dataset._mmap.close()

    def _record_arrow_transform(self, reports: dict[str, dict[str, object]]) -> None:
        dataset = Dataset.from_dict(
            {"x": list(range(16)), "y": [value / 2 for value in range(16)]}
        )
        loader = DataLoader(dataset, batch_size=4, num_workers=1, seed=281)
        try:
            next(iter(loader))
            self._capture(loader, reports, expected_delivery="pinned-transform")
        finally:
            loader.close()

    def _record_parquet_decode(
        self, root: Path, reports: dict[str, dict[str, object]]
    ) -> None:
        path = root / "rows.parquet"
        parquet.write_table(
            pa.table({"x": list(range(16)), "y": [value / 2 for value in range(16)]}),
            path,
        )
        dataset = arrow_dataset.dataset(path, format="parquet")
        loader = DataLoader(dataset, batch_size=4, num_workers=1, seed=283)
        try:
            next(iter(loader))
            self._capture(loader, reports, expected_delivery="pinned-decode")
        finally:
            loader.close()

    def _record_pinned_image(self, reports: dict[str, dict[str, object]]) -> None:
        encoded = [
            encode_png((torch.arange(192, dtype=torch.uint8) + offset).reshape(3, 8, 8))
            for offset in range(4)
        ]
        dataset = pipeline(
            Source(encoded, output_type=torch.Tensor),
            Decode(
                forbidden_decoder,
                input_type=torch.Tensor,
                output_type=torch.Tensor,
                codec="png",
                substitute=True,
            ),
            Collate(torch.stack, input_type=torch.Tensor, output_type=torch.Tensor),
        )
        loader = DataLoader(dataset, batch_size=2, num_workers=1, seed=293)
        try:
            batch = next(iter(loader))
            self.assertEqual(batch.untyped_storage().nbytes(), batch.numel())
            self._capture(loader, reports, expected_delivery="single-write")
        finally:
            loader.close()

    def _record_variable_text(self, reports: dict[str, dict[str, object]]) -> None:
        values = [
            torch.arange(length, dtype=torch.int64) for length in (20, 24, 28, 32)
        ]
        dataset = pipeline(
            Source(values, output_type=torch.Tensor),
            Collate(pad_sequence, input_type=torch.Tensor, output_type=torch.Tensor),
        )
        loader = DataLoader(dataset, batch_size=2, num_workers=1, seed=307)
        try:
            batch = next(iter(loader))
            self.assertGreaterEqual(batch.untyped_storage().nbytes(), batch.numel() * 8)
            self._capture(loader, reports, expected_delivery="single-write")
        finally:
            loader.close()

    def _capture(
        self,
        loader: DataLoader,
        reports: dict[str, dict[str, object]],
        *,
        expected_delivery: str,
    ) -> None:
        report = loader.stats()["memory"]
        self.assertEqual(report["delivery"], expected_delivery)
        self.assertGreater(int(report["produced_samples"]), 0)
        self.assertNotIn(report["source_class"], reports)
        reports[str(report["source_class"])] = report


if __name__ == "__main__":
    unittest.main()

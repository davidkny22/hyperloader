"""Installed public gate for fingerprint-driven profile invalidation."""

from __future__ import annotations

import inspect
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.dataset as arrow_dataset
import pyarrow.parquet as parquet
import torch
from datasets import Dataset
from torch.utils.data import TensorDataset

from hyperloader import DataLoader, HyperConfig, _hyperloader
from hyperloader.config import DeterminismConfig, SchedulerConfig
from hyperloader.fingerprint import FingerprintElement, require_fingerprint_match
from hyperloader.fingerprint import dataset as dataset_module

from .fingerprint_invalidation_support import FileDataset


def _omitted_file_inventory(*_args: object, **_kwargs: object):
    """Plant a weak fingerprint that ignores every file property."""
    return [FingerprintElement("dataset.files.omitted", True)]


class FingerprintInvalidationGate(unittest.TestCase):
    """Prove changed contract inputs cannot reuse stale cost observations."""

    def test_installed_product_is_under_the_declared_root(self) -> None:
        expected_root = os.environ.get("HYPERLOADER_EXPECTED_INSTALL_ROOT")
        if expected_root is None:
            self.skipTest("the installed-artifact root is declared by the gate harness")
        root = Path(expected_root).resolve()
        self.assertTrue(Path(_hyperloader.__file__).resolve().is_relative_to(root))
        self.assertTrue(
            Path(inspect.getfile(DataLoader)).resolve().is_relative_to(root)
        )

    def test_file_change_invalidates_profile_and_takes_the_cold_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            sample = dataset_root / "sample.bin"
            sample.write_bytes(b"left")
            config = HyperConfig(
                scheduler=SchedulerConfig(profile_cache=root / "cache")
            )
            mutation = (
                mock.patch.object(
                    dataset_module,
                    "file_elements",
                    _omitted_file_inventory,
                )
                if os.environ.get("HYPERLOADER_FINGERPRINT_MUTATION") == "omit-files"
                else nullcontext()
            )
            with mutation:
                first = DataLoader(
                    FileDataset(dataset_root, ("sample.bin",)),
                    num_workers=0,
                    config=config,
                )
                first._cost_profile.observe(0, 700)
                original = first._fingerprint
                first_path = first._profile_cache_path
                first.close()

                unchanged = DataLoader(
                    FileDataset(dataset_root, ("sample.bin",)),
                    num_workers=0,
                    config=config,
                )
                try:
                    self.assertEqual(unchanged._cost_profile.estimate(0), 700.0)
                    self.assertEqual(unchanged._profile_cache_path, first_path)
                finally:
                    unchanged.close()

                sample.write_bytes(b"longer")
                changed = DataLoader(
                    FileDataset(dataset_root, ("sample.bin",)),
                    num_workers=0,
                    config=config,
                )
                try:
                    self.assertNotEqual(changed._fingerprint.digest, original.digest)
                    self.assertNotEqual(changed._profile_cache_path, first_path)
                    self.assertIsNone(changed._cost_profile.estimate(0))
                    with self.assertRaisesRegex(
                        ValueError,
                        r"fingerprint mismatch at dataset\.files\[0\]\.size: "
                        r"expected 4, found 6",
                    ):
                        require_fingerprint_match(original, changed._fingerprint)
                finally:
                    changed.close()

    def test_content_mode_excludes_mtime_while_strict_mode_hashes_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "sample.bin"
            sample.write_bytes(b"left")
            content_config = HyperConfig(scheduler=SchedulerConfig(profile_cache="off"))
            strict_config = HyperConfig(
                scheduler=SchedulerConfig(profile_cache="off"),
                determinism=DeterminismConfig(fingerprint="strict"),
            )
            content_before = self._fingerprint(
                FileDataset(root, ("sample.bin",)), content_config
            )
            strict_before = self._fingerprint(
                FileDataset(root, ("sample.bin",)), strict_config
            )
            os.utime(sample, (1, 1))
            self.assertEqual(
                self._fingerprint(FileDataset(root, ("sample.bin",)), content_config),
                content_before,
            )

            sample.write_bytes(b"rght")
            self.assertEqual(
                self._fingerprint(FileDataset(root, ("sample.bin",)), content_config),
                content_before,
            )
            self.assertNotEqual(
                self._fingerprint(FileDataset(root, ("sample.bin",)), strict_config),
                strict_before,
            )

    def test_all_launch_dataset_families_change_on_named_mutations(self) -> None:
        config = HyperConfig(scheduler=SchedulerConfig(profile_cache="off"))
        cases = [
            (
                "tensor-shape",
                torch.arange(6, dtype=torch.int32).reshape(3, 2),
                torch.arange(6, dtype=torch.int32).reshape(2, 3),
            ),
            (
                "tensor-dataset-dtype",
                TensorDataset(torch.arange(4, dtype=torch.int32)),
                TensorDataset(torch.arange(4, dtype=torch.int64)),
            ),
            (
                "arrow-rows",
                Dataset.from_dict({"x": [1, 2]}),
                Dataset.from_dict({"x": [1, 2, 3]}),
            ),
        ]
        for name, before, after in cases:
            with self.subTest(name=name):
                self.assertNotEqual(
                    self._fingerprint(before, config), self._fingerprint(after, config)
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_file = root / "first" / "sample.bin"
            second_file = root / "second" / "sample.bin"
            first_file.parent.mkdir()
            second_file.parent.mkdir()
            first_file.write_bytes(b"left")
            second_file.write_bytes(b"longer")
            self.assertNotEqual(
                self._fingerprint(
                    FileDataset(first_file.parent, ("sample.bin",)), config
                ),
                self._fingerprint(
                    FileDataset(second_file.parent, ("sample.bin",)), config
                ),
            )

            first_file.write_bytes(b"12345678")
            second_file.write_bytes(b"123456")
            first_map = np.memmap(first_file, dtype=np.int16, mode="r", shape=(2, 2))
            second_map = np.memmap(second_file, dtype=np.int16, mode="r", shape=(3, 1))
            try:
                self.assertNotEqual(
                    self._fingerprint(first_map, config),
                    self._fingerprint(second_map, config),
                )
            finally:
                first_map._mmap.close()
                second_map._mmap.close()

            first_parquet = root / "first.parquet"
            second_parquet = root / "second.parquet"
            parquet.write_table(pa.table({"x": [1, 2]}), first_parquet)
            parquet.write_table(pa.table({"x": [1, 2, 3]}), second_parquet)
            self.assertNotEqual(
                self._fingerprint(
                    arrow_dataset.dataset(first_parquet, format="parquet"), config
                ),
                self._fingerprint(
                    arrow_dataset.dataset(second_parquet, format="parquet"), config
                ),
            )

    @staticmethod
    def _fingerprint(dataset: object, config: HyperConfig, **kwargs: object) -> str:
        loader = DataLoader(dataset, num_workers=0, config=config, **kwargs)
        try:
            return loader._fingerprint.digest
        finally:
            loader.close()


if __name__ == "__main__":
    unittest.main()

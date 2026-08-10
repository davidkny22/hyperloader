"""Fixtures and equality helpers for installed planner-equivalence checks."""

from __future__ import annotations

import struct
import unittest
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torchvision.datasets.folder import DatasetFolder

from hyperloader import _hyperloader
from hyperloader.planner import StructurePlan


class DelegatingDataset:
    """Expose one indexable object through an unregistered black-box type."""

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]


class ParquetRows:
    """Express PyArrow's logical row operation as a black-box dataset."""

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset
        self.length = dataset.count_rows()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset.take([index]).to_pylist()[0]


def decode_folder_path(path: str) -> int:
    """Decode a fixture token or raise at its stable sentinel."""
    if path == "explode":
        raise ValueError("planner equivalence sentinel")
    return int(path)


def double(value: int) -> int:
    """Apply the fixture's sample transform."""
    return value * 2


def increment(value: int) -> int:
    """Apply the fixture's target transform."""
    return value + 1


class MemoryFolder(DatasetFolder):
    """Provide DatasetFolder's canonical operations without filesystem discovery."""

    def __init__(self, *, exploding: bool = False, length: int = 8) -> None:
        paths = [str(index) for index in range(length)]
        if exploding:
            paths[3] = "explode"
        self.samples = [(path, index + 10) for index, path in enumerate(paths)]
        self.loader = decode_folder_path
        self.transform = double
        self.target_transform = increment


class IncompleteFolder(DatasetFolder):
    """Violate the canonical-field assumption while remaining black-box indexable."""

    def __init__(self) -> None:
        pass

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[str, int]:
        return "refuge", index


def shifted_structure_index(
    self: StructurePlan, root_seed: int, epoch: int, position: int
) -> int:
    """Plant a public-path mapping error by rotating sampler positions."""
    shifted = (position + 1) % self.length
    if not self.shuffle:
        return shifted
    return _hyperloader._permutation_index(root_seed, epoch, self.length, shifted)


def assert_contract_equal(
    test: unittest.TestCase, actual: object, expected: object
) -> None:
    """Apply a recursive bit-exact relation to two public values."""
    test.assertIs(type(actual), type(expected))
    if isinstance(actual, torch.Tensor):
        test.assertEqual(actual.dtype, expected.dtype)
        test.assertEqual(actual.shape, expected.shape)
        test.assertEqual(actual.layout, expected.layout)
        test.assertEqual(actual.stride(), expected.stride())
        actual_bits = actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        expected_bits = (
            expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        )
        test.assertTrue(torch.equal(actual_bits, expected_bits))
        return
    if isinstance(actual, np.ndarray):
        test.assertEqual(actual.dtype, expected.dtype)
        test.assertEqual(actual.shape, expected.shape)
        test.assertEqual(actual.strides, expected.strides)
        test.assertEqual(actual.tobytes(), expected.tobytes())
        return
    if isinstance(actual, np.generic):
        test.assertEqual(actual.dtype, expected.dtype)
        test.assertEqual(actual.tobytes(), expected.tobytes())
        return
    if isinstance(actual, Mapping):
        test.assertEqual(list(actual), list(expected))
        for key in actual:
            assert_contract_equal(test, actual[key], expected[key])
        return
    if isinstance(actual, Sequence) and not isinstance(actual, (str, bytes, bytearray)):
        test.assertEqual(len(actual), len(expected))
        for actual_item, expected_item in zip(actual, expected, strict=True):
            assert_contract_equal(test, actual_item, expected_item)
        return
    if isinstance(actual, float):
        test.assertEqual(struct.pack("!d", actual), struct.pack("!d", expected))
        return
    test.assertEqual(actual, expected)

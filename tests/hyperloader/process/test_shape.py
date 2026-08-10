"""Probe-derived nested batch-shape tests."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from hyperloader.process.shape import batch_shape


class BatchShapeTest(unittest.TestCase):
    """Exercise default-collation shape inference without materializing a batch."""

    def test_tensor_and_numpy_values_gain_the_batch_dimension(self) -> None:
        tensor = torch.zeros((2, 3), dtype=torch.float32)
        array = np.zeros((4,), dtype=np.int16)

        self.assertEqual(
            batch_shape(tensor, 8),
            {
                "dtype": "torch.float32",
                "kind": "tensor",
                "shape": [8, 2, 3],
            },
        )
        self.assertEqual(
            batch_shape(array, 8),
            {"dtype": "torch.int16", "kind": "tensor", "shape": [8, 4]},
        )

    def test_nested_mapping_preserves_order_and_leaf_dtypes(self) -> None:
        value = {"token": 3, "score": 1.5, "valid": True}

        shape = batch_shape(value, 4)

        self.assertEqual(shape["kind"], "mapping")
        self.assertEqual(
            [item["key"] for item in shape["items"]],
            ["token", "score", "valid"],
        )
        self.assertEqual(shape["items"][0]["value"]["dtype"], "torch.int64")
        self.assertEqual(shape["items"][1]["value"]["dtype"], "torch.float64")
        self.assertEqual(shape["items"][2]["value"]["dtype"], "torch.bool")

    def test_unbatched_numpy_row_retains_its_public_array_shape(self) -> None:
        value = np.zeros((2, 5), dtype=np.float64)

        self.assertEqual(
            batch_shape(value, None),
            {"dtype": "float64", "kind": "ndarray", "shape": [2, 5]},
        )


if __name__ == "__main__":
    unittest.main()

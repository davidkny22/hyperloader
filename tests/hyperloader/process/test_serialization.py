"""Process result storage-identity serialization checks."""

from __future__ import annotations

import pickle
import unittest

import numpy as np
import torch

from hyperloader.process.serialization import (
    MAGIC,
    NUMPY_ARRAY,
    PICKLE_VALUE,
    TENSOR_VIEW,
    ResultDecoder,
    ResultEncoder,
    encode_multiprocessing,
)


class ProcessSerializationTest(unittest.TestCase):
    """Verify tensor caching, view fidelity, and ordinary-object fallback."""

    def test_tensor_storage_crosses_once_then_views_use_coordinates(self) -> None:
        backing = torch.arange(1_048_576, dtype=torch.int64).reshape(-1, 512)
        encoder = ResultEncoder()
        decoder = ResultDecoder()

        first_payload = encoder.encode(backing[0])
        second_payload = encoder.encode(backing[1])
        first = decoder.decode(first_payload, worker=3)
        second = decoder.decode(second_payload, worker=3)

        self.assertLess(len(first_payload), 262_144)
        self.assertEqual(second_payload[len(MAGIC)], TENSOR_VIEW)
        self.assertLess(len(second_payload), 256)
        self.assertTrue(torch.equal(first, backing[0]))
        self.assertTrue(torch.equal(second, backing[1]))
        self.assertEqual(second.stride(), backing[1].stride())

    def test_unknown_tensor_reference_is_rejected(self) -> None:
        encoder = ResultEncoder()
        backing = torch.arange(16).reshape(2, 8)
        encoder.encode(backing[0])
        reference = encoder.encode(backing[1])

        with self.assertRaisesRegex(RuntimeError, "storage reference is unknown"):
            ResultDecoder().decode(reference, worker=0)

    def test_ordinary_values_retain_pickle_semantics(self) -> None:
        value = {"items": [1, "two", b"three"]}

        self.assertEqual(ResultDecoder().decode(ResultEncoder().encode(value), 0), value)

    def test_contiguous_numpy_array_uses_typed_raw_payload(self) -> None:
        value = np.arange(24, dtype=np.int64).reshape(3, 8)

        payload = ResultEncoder().encode(value)
        decoded = ResultDecoder().decode(payload, worker=0)

        self.assertEqual(payload[len(MAGIC)], NUMPY_ARRAY)
        self.assertIs(type(decoded), np.ndarray)
        self.assertEqual(decoded.dtype, value.dtype)
        self.assertEqual(decoded.shape, value.shape)
        self.assertTrue(decoded.flags.writeable)
        np.testing.assert_array_equal(decoded, value)

    def test_noncontiguous_numpy_array_retains_pickle_fallback(self) -> None:
        value = np.arange(24, dtype=np.int64).reshape(3, 8)[:, ::2]

        payload = ResultEncoder().encode(value)
        decoded = ResultDecoder().decode(payload, worker=0)

        self.assertEqual(payload[len(MAGIC)], PICKLE_VALUE)
        self.assertIs(type(decoded), np.ndarray)
        np.testing.assert_array_equal(decoded, value)

    def test_numpy_payload_size_mismatch_is_rejected(self) -> None:
        payload = ResultEncoder().encode(np.arange(8, dtype=np.int64))

        with self.assertRaisesRegex(RuntimeError, "payload size is invalid"):
            ResultDecoder().decode(payload[:-1], worker=0)

    def test_dataset_reducer_creates_independent_transfer_tokens(self) -> None:
        tensor = torch.arange(8)

        first = pickle.loads(encode_multiprocessing(tensor))
        second = pickle.loads(encode_multiprocessing(tensor))

        self.assertTrue(torch.equal(first, tensor))
        self.assertTrue(torch.equal(second, tensor))


if __name__ == "__main__":
    unittest.main()

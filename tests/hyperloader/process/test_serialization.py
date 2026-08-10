"""Process result storage-identity serialization checks."""

from __future__ import annotations

import pickle
import unittest

import torch

from hyperloader.process.serialization import (
    MAGIC,
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

    def test_dataset_reducer_creates_independent_transfer_tokens(self) -> None:
        tensor = torch.arange(8)

        first = pickle.loads(encode_multiprocessing(tensor))
        second = pickle.loads(encode_multiprocessing(tensor))

        self.assertTrue(torch.equal(first, tensor))
        self.assertTrue(torch.equal(second, tensor))


if __name__ == "__main__":
    unittest.main()

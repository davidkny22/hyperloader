"""Canonical Torch compatibility oracle generation and validation."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from benches.compat_golden_cases import generate_cases
from benches.compat_golden_model import (
    FORMAT,
    encode_value,
    read_document,
    validate_document,
    write_document,
)


class CompatGoldenTest(unittest.TestCase):
    """Prove deterministic encoding, schema checks, and semantic coverage."""

    def test_tensor_encoding_preserves_shape_stride_dtype_and_bits(self) -> None:
        value = torch.arange(12, dtype=torch.int64).reshape(3, 4).t()
        encoded = encode_value(value)
        self.assertEqual(encoded["dtype"], "torch.int64")
        self.assertEqual(encoded["shape"], [4, 3])
        self.assertEqual(encoded["stride"], [1, 4])
        self.assertEqual(len(encoded["bits"]), value.numel() * value.element_size() * 2)

    def test_writer_is_canonical_and_digest_backed(self) -> None:
        document = _document({"case": [[encode_value(torch.tensor([1, 2]))]]})
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            first_digest = write_document(first, document)
            second_digest = write_document(second, document)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_digest, second_digest)
            self.assertEqual(
                first_digest, hashlib.sha256(first.read_bytes()).hexdigest()
            )
            self.assertEqual(read_document(first), document)

    def test_schema_rejects_an_unpinned_environment(self) -> None:
        document = _document({"case": [[{"kind": "none"}]]})
        del document["environment"]["torch"]
        with self.assertRaisesRegex(ValueError, "environment fields"):
            validate_document(document)

    def test_real_torch_cases_cover_the_named_contract_surfaces(self) -> None:
        cases = generate_cases()
        self.assertEqual(
            set(cases),
            {
                "zero_worker_shuffle",
                "worker_round_robin",
                "persistent_free_running",
                "worker_initializer",
                "sampler_and_user_collate",
                "iterable_sharding",
            },
        )
        self.assertEqual(len(cases["persistent_free_running"]), 2)
        self.assertTrue(all(epochs for epochs in cases.values()))

    def test_real_torch_cases_repeat_bit_exactly(self) -> None:
        self.assertEqual(generate_cases(), generate_cases())


def _document(cases: dict[str, object]) -> dict[str, object]:
    return {
        "format": FORMAT,
        "environment": {
            "torch": "2.13.0+cpu",
            "torch_minor": "2.13",
            "python": "3.12.6",
            "implementation": "CPython",
            "system": "Windows",
            "release": "11",
            "machine": "AMD64",
            "multiprocessing_start_method": "spawn",
            "in_order": True,
        },
        "cases": cases,
    }


if __name__ == "__main__":
    unittest.main()

"""Canonical Torch compatibility oracle generation and validation."""

from __future__ import annotations

import copy
import hashlib
import platform
import tempfile
import unittest
from collections import namedtuple
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
from benches.generate_compat_golden import build_document, canonical_system
from benches.verify_compat_golden import first_difference, verify_artifact


class CompatGoldenTest(unittest.TestCase):
    """Prove deterministic encoding, schema checks, and semantic coverage."""

    def test_tensor_encoding_preserves_shape_stride_dtype_and_bits(self) -> None:
        value = torch.arange(12, dtype=torch.int64).reshape(3, 4).t()
        encoded = encode_value(value)
        self.assertEqual(encoded["dtype"], "torch.int64")
        self.assertEqual(encoded["shape"], [4, 3])
        self.assertEqual(encoded["stride"], [1, 4])
        self.assertEqual(len(encoded["bits"]), value.numel() * value.element_size() * 2)

    def test_encoder_preserves_container_and_scalar_identity(self) -> None:
        point = namedtuple("Point", ("x", "y"))(1, 2.5)
        encoded = encode_value(
            {"values": [None, True, point, b"x", "y", 2**63]}
        )
        self.assertEqual(encoded["kind"], "mapping")
        items = encoded["items"][0][1]["items"]
        self.assertEqual(
            [item["kind"] for item in items],
            ["none", "bool", "namedtuple", "bytes", "str", "int"],
        )
        self.assertEqual(items[-1]["value"], str(2**63))

    def test_encoder_rejects_an_unsupported_value(self) -> None:
        with self.assertRaisesRegex(TypeError, "unsupported golden value"):
            encode_value(object())

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

    def test_first_difference_names_the_mutated_stream_leaf(self) -> None:
        expected = {"cases": {"worker": [[{"bits": "00"}]]}}
        mutated = {"cases": {"worker": [[{"bits": "01"}]]}}
        self.assertEqual(
            first_difference(expected, mutated),
            "$.cases.worker[0][0].bits",
        )

    def test_pinned_artifact_reproduces_and_stream_mutation_is_red(self) -> None:
        root = Path(__file__).parents[3]
        minor = ".".join(torch.__version__.split("+", 1)[0].split(".")[:2])
        system = platform.system().lower()
        directory = "macos" if system == "darwin" else system
        artifact = root / "oracles" / "torch-golden" / directory / f"torch-{minor}.json"
        if not artifact.is_file():
            self.skipTest("the active Torch and platform pair has no committed oracle")
        report = verify_artifact(artifact)
        self.assertEqual(report["reproduction"], "bit-exact")
        mutated = copy.deepcopy(read_document(artifact))
        self.assertTrue(_mutate_first_bits(mutated))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.json"
            write_document(path, mutated)
            with self.assertRaisesRegex(RuntimeError, r"differs at \$\.cases"):
                verify_artifact(path)

    def test_build_document_rejects_environment_drift_before_generation(self) -> None:
        actual_minor = ".".join(torch.__version__.split("+", 1)[0].split(".")[:2])
        with self.assertRaisesRegex(RuntimeError, "expected Torch 0.0"):
            build_document("0.0", platform.system())
        with self.assertRaisesRegex(RuntimeError, "expected MissingOS"):
            build_document(actual_minor, "MissingOS")

    def test_release_platform_name_accepts_the_darwin_runtime_label(self) -> None:
        self.assertEqual(canonical_system("macos"), "macos")
        self.assertEqual(canonical_system("Darwin"), "macos")
        self.assertEqual(canonical_system("Windows"), "windows")


def _document(cases: dict[str, object]) -> dict[str, object]:
    return {
        "format": FORMAT,
        "environment": {
            "torch": "torch-version-from-record",
            "torch_minor": "torch-minor-from-record",
            "python": "runtime-version-from-record",
            "implementation": "runtime-implementation-from-record",
            "system": "operating-system-from-record",
            "release": "kernel-release-from-record",
            "machine": "architecture-from-record",
            "multiprocessing_start_method": "start-method-from-record",
            "in_order": True,
        },
        "cases": cases,
    }


def _mutate_first_bits(value: object) -> bool:
    if isinstance(value, dict):
        if isinstance(value.get("bits"), str) and value["bits"]:
            value["bits"] = ("1" if value["bits"][0] != "1" else "0") + value["bits"][1:]
            return True
        return any(_mutate_first_bits(item) for item in value.values())
    if isinstance(value, list):
        return any(_mutate_first_bits(item) for item in value)
    return False


if __name__ == "__main__":
    unittest.main()

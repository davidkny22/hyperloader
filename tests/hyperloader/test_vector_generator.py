"""Tests for the sealed contract-vector generator."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

VECTOR_DIR = Path(__file__).parents[2] / "oracles" / "contract-vectors"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reference = _load("contract_vector_reference", VECTOR_DIR / "reference.py")
sys.path.insert(0, str(VECTOR_DIR))
generator = _load("contract_vector_generator", VECTOR_DIR / "generate.py")


class VectorGeneratorTest(unittest.TestCase):
    """Check deterministic generation, frozen bindings, and overwrite safety."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.document = reference.build_document()

    def test_document_covers_every_contract_boundary(self) -> None:
        reference.validate_document(self.document)
        self.assertEqual(len(self.document["philox"]["vectors"]), 900)
        self.assertEqual(
            {vector["domain"] for vector in self.document["permutations"]},
            set(reference.REQUIRED_DOMAINS),
        )

    def test_serialization_is_deterministic(self) -> None:
        self.assertEqual(
            reference.serialize(self.document),
            reference.serialize(reference.build_document()),
        )

    def test_round_word_and_rejection_advancement_are_frozen(self) -> None:
        self.assertEqual(
            self.document["bindings"]["fisher_yates_counter"],
            ["draw_ordinal", 8, 3, 0],
        )
        self.assertTrue(
            self.document["bindings"]["fisher_yates_rejections_advance"]
        )
        self.assertTrue(
            any(
                vector.get("draw_count", 0) > vector["domain"] - 1
                for vector in self.document["permutations"]
                if vector["regime"] == "materialized"
            )
        )

    def test_generator_refuses_to_replace_an_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "vectors.json"
            generator.write_new(output)
            first = output.read_bytes()
            with self.assertRaises(FileExistsError):
                generator.write_new(output)
            self.assertEqual(output.read_bytes(), first)

    def test_schema_rejects_a_non_normative_counter(self) -> None:
        changed = dict(self.document)
        changed["bindings"] = dict(changed["bindings"])
        changed["bindings"]["fisher_yates_counter"] = ["draw_ordinal", 7, 3, 0]
        with self.assertRaisesRegex(ValueError, "counter binding"):
            reference.validate_document(changed)


if __name__ == "__main__":
    unittest.main()

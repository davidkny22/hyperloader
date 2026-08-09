"""Reproduce every committed contract vector through independent implementations."""

from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path

from hyperloader import DataLoader, _hyperloader

from contract_vector_harness import (
    assert_installed_under,
    load_document,
    permutation_digest,
    reference,
)
from placement_reference import rank_placements
from rng_reference import (
    block,
    feistel_permute,
    materialized_permutation,
    philox4x32_10,
)


class ContractVectorTest(unittest.TestCase):
    """Verify artifact schema, native reproduction, and independent reproduction."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.document = load_document()

    def test_installed_package_path_and_public_loader_wiring(self) -> None:
        expected_root = os.environ.get("HYPERLOADER_EXPECTED_INSTALL_ROOT")
        if expected_root is None:
            self.skipTest("isolated install root is asserted by the gate runner")
        assert_installed_under(_hyperloader.__file__, expected_root)
        loader = DataLoader([0, 1], batch_size=1, seed=7)
        self.assertEqual(loader.seed, 7)

    def test_install_root_assertion_rejects_a_foreign_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AssertionError, "outside installed root"):
                assert_installed_under(_hyperloader.__file__, Path(directory))

    def test_random123_known_answer_and_all_philox_vectors(self) -> None:
        self.assertEqual(
            list(reference.philox4x32_10((0, 0, 0, 0), (0, 0))),
            self.document["philox"]["random123_zero"],
        )
        self.assertEqual(
            list(philox4x32_10((0, 0, 0, 0), (0, 0))),
            self.document["philox"]["random123_zero"],
        )
        for vector in self.document["philox"]["vectors"]:
            arguments = (
                vector["root_seed"],
                vector["epoch"],
                vector["coord"],
                vector["draw_index"],
                vector["stream_id"],
            )
            with self.subTest(arguments=arguments):
                self.assertEqual(list(_hyperloader._rng_block(*arguments)), vector["words"])
                self.assertEqual(list(block(*arguments)), vector["words"])

    def test_all_permutation_vectors(self) -> None:
        for vector in self.document["permutations"]:
            root_seed = vector["root_seed"]
            epoch = vector["epoch"]
            domain = vector["domain"]
            with self.subTest(root_seed=root_seed, epoch=epoch, domain=domain):
                if vector["regime"] == "materialized":
                    native = list(
                        _hyperloader._materialized_permutation(
                            root_seed, epoch, domain
                        )
                    )
                    independent, draws = materialized_permutation(
                        root_seed, epoch, domain
                    )
                    self.assertEqual(permutation_digest(native), vector["digest"])
                    self.assertEqual(permutation_digest(independent), vector["digest"])
                    self.assertEqual(draws, vector["draw_count"])
                    for position, expected in vector["points"]:
                        self.assertEqual(native[position], expected)
                else:
                    for position, expected in vector["points"]:
                        arguments = (root_seed, epoch, domain, position)
                        self.assertEqual(
                            _hyperloader._feistel_permute(*arguments), expected
                        )
                        self.assertEqual(feistel_permute(*arguments), expected)

    def test_all_placement_vectors(self) -> None:
        for vector in self.document["placements"]:
            for rank_vector in vector["ranks"]:
                arguments = (
                    vector["root_seed"],
                    vector["epoch"],
                    vector["dataset_len"],
                    vector["batch_size"],
                    vector["world_size"],
                    rank_vector["rank"],
                    vector["drop_last"],
                    vector["exact_count"],
                )
                expected = [tuple(item) for item in rank_vector["items"]]
                with self.subTest(name=vector["name"], rank=rank_vector["rank"]):
                    self.assertEqual(_hyperloader._rank_placements(*arguments), expected)
                    self.assertEqual(rank_placements(*arguments), expected)

    def test_schema_rejects_missing_edge_domain(self) -> None:
        changed = copy.deepcopy(self.document)
        changed["permutations"] = [
            vector
            for vector in changed["permutations"]
            if vector["domain"] != 1_000_000_007
        ]
        with self.assertRaisesRegex(ValueError, "required domain"):
            reference.validate_document(changed)

    def test_schema_rejects_changed_round_word(self) -> None:
        changed = copy.deepcopy(self.document)
        changed["bindings"]["fisher_yates_counter"][1] = 7
        with self.assertRaisesRegex(ValueError, "counter binding"):
            reference.validate_document(changed)


if __name__ == "__main__":
    unittest.main()

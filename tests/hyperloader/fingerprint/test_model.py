"""Ordered fingerprint representation and diagnostic tests."""

from __future__ import annotations

import unittest

from hyperloader.fingerprint import (
    ContractFingerprint,
    FingerprintElement,
    require_fingerprint_match,
)


class FingerprintModelTest(unittest.TestCase):
    """Exercise canonical digests, state payloads, and first-change errors."""

    def test_round_trip_preserves_digest_and_order(self) -> None:
        fingerprint = ContractFingerprint(
            (
                FingerprintElement("contract_version", 1),
                FingerprintElement("dataset.shape", [4, 2]),
            )
        )

        restored = ContractFingerprint.from_dict(fingerprint.to_dict())

        self.assertEqual(restored, fingerprint)
        self.assertEqual(restored.digest, fingerprint.digest)

    def test_first_changed_element_is_named_exactly(self) -> None:
        expected = ContractFingerprint(
            (
                FingerprintElement("contract_version", 1),
                FingerprintElement("dataset.files[0].size", 4),
                FingerprintElement("delivery", "in-order"),
            )
        )
        actual = ContractFingerprint(
            (
                FingerprintElement("contract_version", 1),
                FingerprintElement("dataset.files[0].size", 5),
                FingerprintElement("delivery", "on-completion"),
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            r"fingerprint mismatch at dataset\.files\[0\]\.size: expected 4, found 5",
        ):
            require_fingerprint_match(expected, actual)

    def test_digest_claim_is_recomputed_on_restore(self) -> None:
        payload = {
            "digest": "0" * 64,
            "elements": [{"path": "contract_version", "value": 1}],
        }

        with self.assertRaisesRegex(ValueError, "digest"):
            ContractFingerprint.from_dict(payload)


if __name__ == "__main__":
    unittest.main()

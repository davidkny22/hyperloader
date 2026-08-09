"""Byte-split checks for the paired overhead protocol."""

from __future__ import annotations

import unittest

from benches.benchmark_protocol import process_transport_split


class ProcessTransportAccountingTest(unittest.TestCase):
    """Verify exact attribution and reject inconsistent observations."""

    def test_separates_permitted_writes_from_explicit_transport_overhead(self) -> None:
        split = process_transport_split(
            duration_seconds=2.0,
            samples=8,
            batches=2,
            logical_sample_bytes=4,
            serialized_sample_bytes=5,
            batch_bytes=16,
        )

        self.assertEqual(split.model_input_gbps, 16 / 1_000_000_000)
        self.assertEqual(split.irreducible_host_gbps, 32 / 1_000_000_000)
        self.assertEqual(split.explicit_overhead_gbps, 44 / 1_000_000_000)
        self.assertEqual(split.explicit_total_host_gbps, 76 / 1_000_000_000)

    def test_rejects_unbalanced_or_impossible_copy_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "balance"):
            process_transport_split(
                duration_seconds=1.0,
                samples=3,
                batches=1,
                logical_sample_bytes=4,
                serialized_sample_bytes=5,
                batch_bytes=16,
            )
        with self.assertRaisesRegex(ValueError, "smaller"):
            process_transport_split(
                duration_seconds=1.0,
                samples=4,
                batches=1,
                logical_sample_bytes=4,
                serialized_sample_bytes=3,
                batch_bytes=16,
            )


if __name__ == "__main__":
    unittest.main()

"""Fixed-text delivery-memory diagnostic helpers."""

from __future__ import annotations

import importlib
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

BENCHES = Path(__file__).parents[3] / "benches"
sys.path.insert(0, str(BENCHES))
diagnostic = importlib.import_module("dominance_tensor_diagnostic")


class DominanceTensorDiagnosticTest(unittest.TestCase):
    def test_batch_description_names_layout_and_ownership(self) -> None:
        base = torch.arange(48, dtype=torch.int64).reshape(6, 8)
        batch = base[2:4]

        description = diagnostic.describe_batch(batch)

        self.assertEqual(description["shape"], [2, 8])
        self.assertEqual(description["stride"], [8, 1])
        self.assertEqual(description["storage_offset"], 16)
        self.assertEqual(description["logical_bytes"], 128)
        self.assertGreater(description["storage_bytes"], description["logical_bytes"])
        self.assertTrue(description["contiguous"])
        self.assertFalse(description["pinned"])

    def test_variants_preserve_values_and_separate_storage(self) -> None:
        hyper = [torch.arange(16, dtype=torch.int64).reshape(2, 8)]
        reference = [hyper[0].clone().share_memory_()]

        variants = diagnostic.build_variants(hyper, reference)

        for batches in variants.values():
            self.assertTrue(torch.equal(batches[0], hyper[0]))
        self.assertEqual(variants["hyper-view"][0].data_ptr(), hyper[0].data_ptr())
        self.assertNotEqual(variants["hyper-clone"][0].data_ptr(), hyper[0].data_ptr())
        self.assertTrue(variants["hyper-shared-clone"][0].is_shared())
        self.assertTrue(variants["torch-shared"][0].is_shared())

    def test_prefetched_measurement_retains_identity_values(self) -> None:
        batches = [torch.arange(8, dtype=torch.int64)]

        class Feeder:
            def next_batch(self) -> torch.Tensor:
                return batches[0]

        class Workload:
            seen = 0

            def run(self, batch: torch.Tensor) -> None:
                self.seen += int(torch.equal(batch, batches[0]))

        workload = Workload()
        for writeback in (False, True):
            with self.subTest(writeback=writeback):
                workload.seen = 0
                with ThreadPoolExecutor(max_workers=1) as executor:
                    report = diagnostic.measure_prefetched(
                        workload,
                        Feeder(),
                        executor,
                        0.005,
                        writeback=writeback,
                    )

                self.assertGreater(report["iterations"], 0)
                self.assertEqual(workload.seen, report["iterations"])
                self.assertTrue(torch.equal(batches[0], torch.arange(8)))
                self.assertGreaterEqual(report["wait_seconds_per_iteration"], 0.0)


if __name__ == "__main__":
    unittest.main()

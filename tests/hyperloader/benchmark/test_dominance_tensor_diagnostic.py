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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA host allocator is required")
    def test_pinned_clone_bank_preserves_values(self) -> None:
        batches = [torch.arange(16, dtype=torch.int64).reshape(2, 8)]

        pinned = diagnostic.build_pinned_clone_bank(batches)

        self.assertTrue(torch.equal(pinned[0], batches[0]))
        self.assertTrue(pinned[0].is_pinned())
        self.assertNotEqual(pinned[0].data_ptr(), batches[0].data_ptr())

    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA host registration is required"
    )
    def test_host_registration_pins_distinct_source_storage_once(self) -> None:
        source = torch.arange(32, dtype=torch.int64)
        batches = [source[:16], source[16:]]

        with diagnostic.RegisteredHostStorages(batches) as registration:
            self.assertEqual(registration.storage_count, 1)
            self.assertEqual(
                registration.total_bytes, source.untyped_storage().nbytes()
            )
            self.assertTrue(all(batch.is_pinned() for batch in batches))

        self.assertTrue(all(not batch.is_pinned() for batch in batches))

    def test_writeback_traffic_records_stores_and_preserves_values(self) -> None:
        batch = torch.arange(32 * 512, dtype=torch.int64).reshape(32, 512)

        report = diagnostic.measure_writeback_traffic(batch, iterations=16)

        self.assertEqual(report["version_delta"], 16)
        self.assertTrue(report["values_preserved"])
        self.assertEqual(report["logical_bytes_per_iteration"], batch.nbytes)
        self.assertGreater(report["add_seconds_per_iteration"], 0.0)
        self.assertGreater(report["add_read_write_gb_per_second"], 0.0)
        self.assertGreater(report["copy_read_write_gb_per_second"], 0.0)

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
                self.assertGreater(report["preparation_seconds_per_iteration"], 0.0)
                self.assertGreater(report["preparation_effective_gb_per_second"], 0.0)
                self.assertEqual(
                    report["version_delta_per_iteration"], 1.0 if writeback else 0.0
                )


if __name__ == "__main__":
    unittest.main()

"""Spark hardware assurance for calibrated machine and delivery controls."""

from __future__ import annotations

import os
import time
import unittest

import torch
from hyperloader import DataLoader


@unittest.skipUnless(
    os.environ.get("HYPERLOADER_SPARK_HARDWARE") == "1",
    "requires the explicit Spark hardware assurance environment",
)
class SparkMachineDeliveryTest(unittest.TestCase):
    """Exercise calibrated controls through the installed public loader."""

    def test_auto_controls_preserve_fixed_text_storage_and_values(self) -> None:
        source = torch.arange(2048 * 512, dtype=torch.int64).reshape(2048, 512)
        source_version = source._version
        loader = DataLoader(source, batch_size=64, num_workers=1)
        try:
            iterator = iter(loader)
            delivered = []
            snapshot = loader.stats()
            for batch_index in range(32):
                batch = next(iterator)
                delivered.append(batch)
                self.assertEqual(
                    batch.untyped_storage().data_ptr(),
                    source.untyped_storage().data_ptr(),
                )
                self.assertTrue(
                    torch.equal(batch, source[batch_index * 64 : (batch_index + 1) * 64])
                )
                time.sleep(0.003)
                snapshot = loader.stats()
                if (
                    len(delivered) >= 3
                    and snapshot["current"]["machine_keeping_duty"] > 0.0
                ):
                    break

            self.assertEqual(loader.delivery_memory, "pinned")
            self.assertEqual(source._version, source_version)
            self.assertEqual(
                snapshot["memory"].get("pinned_registered_bytes", 0),
                source.numel() * source.element_size(),
            )
            self.assertEqual(snapshot["memory"].get("pinned_staged_bytes", 0), 0)
            self.assertGreater(snapshot["current"]["machine_keeping_duty"], 0.0)
            self.assertLessEqual(snapshot["current"]["machine_keeping_duty"], 0.05)
        finally:
            loader.close()


if __name__ == "__main__":
    unittest.main()

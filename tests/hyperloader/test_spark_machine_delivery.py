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
            first = next(iterator)
            time.sleep(0.003)
            second = next(iterator)
            snapshot = loader.stats()

            self.assertEqual(loader.delivery_memory, "host")
            self.assertEqual(source._version, source_version)
            self.assertEqual(
                first.untyped_storage().data_ptr(), source.untyped_storage().data_ptr()
            )
            self.assertEqual(
                second.untyped_storage().data_ptr(), source.untyped_storage().data_ptr()
            )
            self.assertTrue(torch.equal(first, source[:64]))
            self.assertTrue(torch.equal(second, source[64:128]))
            self.assertEqual(snapshot["memory"].get("pinned_registered_bytes", 0), 0)
            self.assertEqual(snapshot["memory"].get("pinned_staged_bytes", 0), 0)
            self.assertGreater(snapshot["current"]["machine_keeping_duty"], 0.0)
            self.assertLessEqual(snapshot["current"]["machine_keeping_duty"], 0.05)
        finally:
            loader.close()


if __name__ == "__main__":
    unittest.main()

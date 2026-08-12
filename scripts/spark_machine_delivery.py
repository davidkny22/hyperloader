"""Spark hardware assurance for calibrated machine and delivery controls."""

from __future__ import annotations

import importlib.metadata
import time
import unittest

import torch
from torchvision.io import decode_png, encode_png

from hyperloader import Collate, DataLoader, Decode, HyperConfig, Source, pipeline
from hyperloader.config import DeterminismConfig


def forbidden_decode(_value: torch.Tensor) -> torch.Tensor:
    """Fail when the selected Spark decoder does not replace the refuge callable."""
    raise AssertionError("selected decoder did not execute")


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
                    torch.equal(
                        batch, source[batch_index * 64 : (batch_index + 1) * 64]
                    )
                )
                batch.to("cuda", non_blocking=True)
                torch.cuda.synchronize()
                time.sleep(0.08)
                snapshot = loader.stats()
                if (
                    len(delivered) >= 3
                    and snapshot["current"]["machine_keeping_duty"] > 0.0
                    and 0 in set(loader._machine_keeper.cpus())
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
            keeper_cpus = set(loader._machine_keeper.cpus())
            self.assertIn(0, keeper_cpus)
            self.assertIn(loader._machine_keeper_consumer_cpu, keeper_cpus)
        finally:
            loader.close()

    def test_auto_pinned_image_stack_is_the_final_delivery_buffer(self) -> None:
        encoded = [
            encode_png(
                (torch.arange(3072, dtype=torch.uint8) + offset).reshape(3, 32, 32)
            )
            for offset in range(8)
        ]
        dataset = pipeline(
            Source(encoded, output_type=torch.Tensor),
            Decode(
                forbidden_decode,
                input_type=torch.Tensor,
                output_type=torch.Tensor,
                codec="png",
                substitute=True,
            ),
            Collate(torch.stack, input_type=torch.Tensor, output_type=torch.Tensor),
        )
        version = importlib.metadata.version("torchvision").split("+", 1)[0]
        loader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=2,
            config=HyperConfig(
                determinism=DeterminismConfig(
                    decoder_pins={
                        "pipeline-decode-0": f"torchvision.io.decode_png@{version}"
                    }
                )
            ),
        )
        try:
            batches = list(loader)
            report = loader.stats()["memory"]
        finally:
            loader.close()

        self.assertEqual(len(batches), 2)
        self.assertTrue(all(batch.is_pinned() for batch in batches))
        self.assertTrue(
            torch.equal(
                batches[0], torch.stack([decode_png(value) for value in encoded[:4]])
            )
        )
        self.assertTrue(
            torch.equal(
                batches[1], torch.stack([decode_png(value) for value in encoded[4:]])
            )
        )
        self.assertEqual(report["delivery_memory"], "pinned")
        self.assertEqual(report["pinned_registered_bytes"], 0)
        self.assertEqual(report["pinned_staged_bytes"], 0)
        self.assertEqual(report["bytes_beyond_irreducible"], 0)


if __name__ == "__main__":
    unittest.main()

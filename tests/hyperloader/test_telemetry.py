"""Public telemetry snapshots across enabled and disabled execution paths."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from hyperloader import DataLoader, HyperConfig
from hyperloader.config import ControlConfig, TelemetryConfig


class TelemetryPublicTest(unittest.TestCase):
    """Verify delivery tails, epoch summaries, decisions, and disabled behavior."""

    def test_process_epoch_publishes_delivery_and_controller_summary(self) -> None:
        config = HyperConfig(control=ControlConfig(cadence=1e-9))
        loader = DataLoader(
            range(16), batch_size=2, num_workers=2, seed=29, config=config
        )
        try:
            self.assertEqual(sum(batch.numel() for batch in loader), 16)
            snapshot = loader.stats()
        finally:
            loader.close()

        summary = snapshot["last_epoch"]
        latency = summary["delivery_latency_ns"]
        self.assertTrue(snapshot["enabled"])
        self.assertGreater(snapshot["startup_ns"], 0)
        self.assertEqual(summary["delivered_samples"], 16)
        self.assertEqual(summary["delivered_batches"], 8)
        self.assertGreater(summary["delivered_bytes"], 0)
        self.assertGreater(summary["delivery_rate"], 0)
        self.assertLessEqual(latency["p50"], latency["p95"])
        self.assertLessEqual(latency["p95"], latency["p99"])
        self.assertTrue(summary["controller_decisions"])

    def test_tensor_epoch_uses_the_same_public_snapshot(self) -> None:
        dataset = torch.arange(24, dtype=torch.int64).reshape(6, 4)
        loader = DataLoader(dataset, batch_size=2, num_workers=2)
        try:
            self.assertEqual(sum(batch.shape[0] for batch in loader), 6)
            summary = loader.stats()["last_epoch"]
        finally:
            loader.close()

        self.assertEqual(summary["delivered_samples"], 6)
        self.assertEqual(summary["delivered_batches"], 3)
        self.assertEqual(summary["delivered_bytes"], dataset.numel() * dataset.element_size())

    def test_current_snapshot_flushes_a_partial_delivery_group(self) -> None:
        loader = DataLoader(torch.arange(32), batch_size=2, num_workers=2)
        iterator = iter(loader)
        try:
            self.assertEqual(next(iterator).tolist(), [0, 1])
            current = loader.stats()["current"]
            self.assertEqual(current["delivered_samples"], 2)
            self.assertEqual(current["delivered_batches"], 1)
            self.assertEqual(list(iterator)[-1].tolist(), [30, 31])
        finally:
            loader.close()

    def test_disabled_configuration_allocates_no_native_recorder(self) -> None:
        config = HyperConfig(telemetry=TelemetryConfig(enabled=False))
        with mock.patch(
            "hyperloader.telemetry.runtime._hyperloader._Telemetry",
            side_effect=AssertionError("disabled telemetry allocated a recorder"),
        ):
            loader = DataLoader(torch.arange(8), batch_size=2, num_workers=2, config=config)
        try:
            self.assertEqual(sum(batch.numel() for batch in loader), 8)
            snapshot = loader.stats()
        finally:
            loader.close()

        self.assertFalse(snapshot["enabled"])
        self.assertIsNone(snapshot["current"])
        self.assertIsNone(snapshot["last_epoch"])


if __name__ == "__main__":
    unittest.main()

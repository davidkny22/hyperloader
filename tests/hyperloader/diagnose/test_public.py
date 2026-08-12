"""Public passive and active loader diagnosis behavior."""

from __future__ import annotations

import json
import time
import unittest

import torch
from hyperloader import DataLoader, diagnose


class CountingTorchLoader(torch.utils.data.DataLoader):
    """Expose whether passive diagnosis attempted to start iteration."""

    iterations = 0

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.iterations += 1
        return super().__iter__()


class SleepingDataset(torch.utils.data.Dataset):
    """Release the interpreter lane while the active probe fetches a sample."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> int:
        time.sleep(0.01)
        return index


class DiagnosisPublicTest(unittest.TestCase):
    """Exercise both public loader families without private test seams."""

    def test_stock_snapshot_is_passive_and_machine_readable(self) -> None:
        loader = CountingTorchLoader(range(8), batch_size=2, num_workers=0)
        loader.iterations = 0

        report = diagnose(loader)

        self.assertEqual(loader.iterations, 0)
        self.assertEqual(report.record["loader_kind"], "torch")
        self.assertEqual(report.record["observation_mode"], "passive")
        self.assertIsNone(report.record["blocking"]["fraction"])
        self.assertIn("Loader diagnosis", str(report))
        json.dumps(report.record)
        copied = report.to_dict()
        copied["loader_kind"] = "changed"
        self.assertEqual(report.record["loader_kind"], "torch")

    def test_native_snapshot_does_not_advance_the_live_iterator(self) -> None:
        loader = DataLoader(range(8), batch_size=2, num_workers=2, seed=613)
        iterator = iter(loader)
        try:
            self.assertEqual(next(iterator).tolist(), [0, 1])
            before = loader.state_dict()

            report = diagnose(loader)

            after = loader.state_dict()
            self.assertEqual(before, after)
            self.assertEqual(next(iterator).tolist(), [2, 3])
            self.assertEqual(report.record["loader_kind"], "hyperloader")
            self.assertEqual(report.record["observation_mode"], "passive")
            self.assertEqual(len(report.record["workers"]), 2)
            for worker in report.record["workers"]:
                self.assertTrue(worker["alive"])
                self.assertGreaterEqual(worker["cpu_ns"], 0)
                self.assertGreater(worker["rss_bytes"], 0)
            self.assertIsNotNone(report.record["blocking"]["fraction"])
            self.assertIsNone(
                report.record["promotion"]["expected_gain"]["lower_percent"]
            )
            self.assertIn("cause", report.record["attribution"])
        finally:
            loader.close()

    def test_active_probe_reports_its_consumption_and_cost(self) -> None:
        loader = torch.utils.data.DataLoader(
            SleepingDataset(), batch_size=1, num_workers=0
        )

        report = diagnose(loader, probe=True, probe_batches=2)

        probe = report.record["probe"]
        self.assertEqual(report.record["observation_mode"], "active-probe")
        self.assertEqual(probe["consumed_batches"], 2)
        self.assertGreater(probe["elapsed_ns"], 0)
        self.assertGreater(probe["gil_release_fraction"], 0.5)
        self.assertEqual(
            report.record["gil_release"]["fraction"],
            probe["gil_release_fraction"],
        )
        self.assertIn("2 batches consumed", report.text)

    def test_stock_workers_expose_queue_and_process_observations(self) -> None:
        loader = torch.utils.data.DataLoader(
            range(12),
            batch_size=2,
            num_workers=1,
            persistent_workers=True,
            prefetch_factor=2,
        )
        iterator = iter(loader)
        try:
            next(iterator)
            received = iterator._rcvd_idx

            report = diagnose(loader)

            self.assertEqual(iterator._rcvd_idx, received)
            self.assertEqual(len(report.record["workers"]), 1)
            self.assertEqual(report.record["saturation"]["capacity_batches"], 2)
            self.assertIn(
                report.record["saturation"]["ready_batches"], range(3)
            )
            self.assertTrue(report.record["workers"][0]["alive"])
            self.assertGreater(report.record["workers"][0]["rss_bytes"], 0)
        finally:
            iterator._shutdown_workers()

    def test_active_probe_rejects_live_or_invalid_requests(self) -> None:
        loader = torch.utils.data.DataLoader(
            range(8), batch_size=2, num_workers=1, persistent_workers=True
        )
        iterator = iter(loader)
        try:
            with self.assertRaisesRegex(RuntimeError, "no live iterator"):
                diagnose(loader, probe=True)
        finally:
            iterator._shutdown_workers()

        idle = torch.utils.data.DataLoader(range(2), batch_size=1)
        for invalid in (0, 33, True):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "1 through 32"
            ):
                diagnose(idle, probe=True, probe_batches=invalid)

    def test_unknown_object_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "torch DataLoader"):
            diagnose(object())


if __name__ == "__main__":
    unittest.main()

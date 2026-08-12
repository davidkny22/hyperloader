"""Installed public gate for bounded adaptive frontier behavior."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

from hyperloader import DataLoader, HyperConfig, _hyperloader
from hyperloader.config import SchedulerConfig
from hyperloader.process import iterator as iterator_module


class PathologicalCostDataset:
    """Expose a repeatable heavy tail across otherwise inexpensive samples."""

    def __len__(self) -> int:
        return 64

    def __getitem__(self, index: int) -> int:
        time.sleep(0.008 if index in {11, 26, 41, 56} else 0.00025)
        return index


def _unbounded_depth(loader: DataLoader) -> int:
    """Plant a missing ceiling clamp in the public iterator path."""
    return loader._process_pool.frontier_ceiling * 2


class FrontierBudgetGate(unittest.TestCase):
    """Prove formula bounds, adapted stalls, and named binding causes."""

    @unittest.skipUnless(sys.platform == "win32", "timing floor is pinned on Windows")
    def test_pathological_costs_stay_bounded_after_adaptation(self) -> None:
        expected_root = os.environ.get("HYPERLOADER_EXPECTED_INSTALL_ROOT")
        if expected_root is not None:
            self.assertTrue(
                Path(_hyperloader.__file__)
                .resolve()
                .is_relative_to(Path(expected_root).resolve())
            )
        with tempfile.TemporaryDirectory() as directory:
            loader = DataLoader(
                PathologicalCostDataset(),
                batch_size=1,
                num_workers=4,
                seed=101,
                config=HyperConfig(
                    scheduler=SchedulerConfig(profile_cache=Path(directory))
                ),
            )
            mutation = (
                mock.patch.object(iterator_module, "frontier_depth", _unbounded_depth)
                if os.environ.get("HYPERLOADER_FRONTIER_MUTATION") == "ignore-ceiling"
                else nullcontext()
            )
            try:
                self.assertEqual(
                    [int(value.item()) for value in loader], list(range(64))
                )
                with mutation:
                    started = time.perf_counter_ns()
                    delivered = []
                    for value in loader:
                        delivered.append(int(value.item()))
                        time.sleep(0.004)
                    elapsed_ns = time.perf_counter_ns() - started
                report = loader._last_frontier_report
            finally:
                loader.close()

        stall_fraction = report["wait_ns"] / elapsed_ns
        self.assertEqual(delivered, list(range(64)))
        self.assertLessEqual(report["max_occupied"], report["ceiling"])
        self.assertLessEqual(stall_fraction, 0.001)
        self.assertEqual(report["binding"], "profile-tail")
        self._write_metrics(
            {
                **report,
                "elapsed_ns": elapsed_ns,
                "stall_fraction_with_consumer": stall_fraction,
            }
        )

    def test_budget_below_liveness_floor_keeps_two_batches(self) -> None:
        config = HyperConfig(scheduler=SchedulerConfig(frontier_budget=1))
        loader = DataLoader(range(8), batch_size=2, num_workers=2, config=config)
        try:
            self.assertEqual(loader._process_pool.frontier_ceiling, 4)
            self.assertTrue(loader._process_pool.frontier_budget_bound)
            self.assertEqual(sum(batch.numel() for batch in loader), 8)
        finally:
            loader.close()

    def test_profile_cache_off_retains_the_cold_rule(self) -> None:
        config = HyperConfig(scheduler=SchedulerConfig(profile_cache="off"))
        loader = DataLoader(range(8), batch_size=2, num_workers=2, config=config)
        try:
            self.assertIsNone(loader._cost_profile)
            self.assertEqual(
                [value.tolist() for value in loader], [[0, 1], [2, 3], [4, 5], [6, 7]]
            )
            self.assertEqual(loader._last_frontier_report["binding"], "cold-variance")
        finally:
            loader.close()

    @staticmethod
    def _write_metrics(metrics: dict[str, int | float | str]) -> None:
        path = os.environ.get("HYPERLOADER_FRONTIER_METRICS")
        if path is not None:
            Path(path).write_text(
                json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    unittest.main()

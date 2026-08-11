"""Host attribution and consumer-profile helper tests."""

from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BENCHES = Path(__file__).parents[3] / "benches"
sys.path.insert(0, str(BENCHES))
thread_activity = importlib.import_module("dominance_thread_activity")
pyspy_analyze = importlib.import_module("dominance_pyspy_analyze")


class DominanceHostAttributionTest(unittest.TestCase):
    def test_proc_stat_parser_handles_thread_names_with_spaces(self) -> None:
        fields = ["S"] + ["0"] * 10 + ["17", "5"] + ["0"] * 30
        record = "123 (adapter worker 1) " + " ".join(fields)

        parsed = thread_activity._parse_task_stat(record)

        self.assertEqual(parsed["kernel_name"], "adapter worker 1")
        self.assertEqual(parsed["user_ticks"], 17)
        self.assertEqual(parsed["system_ticks"], 5)
        self.assertEqual(parsed["cpu_ticks"], 22)

    def test_thread_cpu_diff_reports_existing_and_created_tasks(self) -> None:
        before = {
            10: {
                "task_id": 10,
                "kernel_name": "python3",
                "python_name": "MainThread",
                "cpu_ticks": 100,
            }
        }
        after = {
            10: {
                "task_id": 10,
                "kernel_name": "python3",
                "python_name": "MainThread",
                "cpu_ticks": 125,
            },
            11: {
                "task_id": 11,
                "kernel_name": "adapter",
                "python_name": "adapter",
                "cpu_ticks": 5,
            },
        }

        with mock.patch.object(
            thread_activity.os, "sysconf", return_value=100, create=True
        ):
            rows = thread_activity.diff_thread_cpu(before, after)

        self.assertEqual(rows[0]["cpu_milliseconds"], 250.0)
        self.assertFalse(rows[0]["created_during_phase"])
        self.assertTrue(rows[1]["created_during_phase"])

    def test_pyspy_analyzer_splits_halves_stages_and_threads(self) -> None:
        raw = (
            "MainThread;profile_hyperloader_half;_profile_half;profile_sync 3\n"
            "adapter;profile_hyperloader_half;_profile_half;profile_next_batch 2\n"
            "MainThread;profile_torch_half;_profile_half;profile_launch 4\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.raw"
            path.write_text(raw, encoding="utf-8")
            report = pyspy_analyze.analyze(path)

        hyper = report["halves"]["hyperloader"]
        torch = report["halves"]["torch"]
        self.assertEqual(hyper["samples"], 5)
        self.assertEqual(torch["samples"], 4)
        self.assertEqual(hyper["stage_samples"][0]["name"], "sync")
        self.assertEqual(hyper["thread_samples"][0]["name"], "MainThread")

    def test_pyspy_analyzer_attributes_all_single_system_threads(self) -> None:
        raw = (
            "thread (1): MainThread;profile_sync;wait 4\n"
            "thread (2): feeder;adapter_work 3\n"
            "thread (3): alu-spinner-17;spin 9\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.raw"
            path.write_text(raw, encoding="utf-8")
            report = pyspy_analyze.analyze(path, "hyperloader")

        half = report["halves"]["hyperloader"]
        self.assertEqual(half["samples"], 7)
        self.assertEqual(half["excluded_auxiliary_samples"], 9)
        self.assertEqual(
            half["thread_samples"],
            [
                {
                    "name": "thread (1): MainThread",
                    "samples": 4,
                    "percent": 400 / 7,
                },
                {
                    "name": "thread (2): feeder",
                    "samples": 3,
                    "percent": 300 / 7,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()

"""Safety checks for temporary Spark clock control."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import call, patch

from benches.spark_clock_guard import run_guard


class SparkClockGuardTest(unittest.TestCase):
    """Require evidence and reset even when the benchmark command fails."""

    def test_command_failure_still_restores_clock_and_writes_evidence(self) -> None:
        no_processes = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        success = subprocess.CompletedProcess([], 0, stdout="All done.\n", stderr="")
        failed = subprocess.CalledProcessError(7, ["benchmark"])
        with TemporaryDirectory() as directory:
            evidence = Path(directory) / "clock.json"
            with patch(
                "benches.spark_clock_guard._run",
                side_effect=[no_processes, success, success],
            ) as control, patch(
                "benches.spark_clock_guard.subprocess.run", side_effect=failed
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    run_guard(
                        evidence=evidence,
                        clock_mhz=2400,
                        command=["benchmark"],
                    )

            self.assertTrue(evidence.is_file())
            self.assertIn('"command_returncode": 7', evidence.read_text())
            self.assertEqual(
                control.call_args_list[-1],
                call(["sudo", "-n", "nvidia-smi", "-rgc"]),
            )


if __name__ == "__main__":
    unittest.main()

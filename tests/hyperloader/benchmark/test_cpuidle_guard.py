"""Safety checks for scoped Spark CPU-idle state control."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from benches.spark_cpuidle_guard import run_guard


class SparkCpuIdleGuardTest(unittest.TestCase):
    """Require exact targets, evidence, and restoration on command failure."""

    def test_command_failure_restores_every_deep_state_to_zero(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "cpu"
            targets = []
            for cpu in (0, 19):
                for state in (0, 1, 2):
                    path = root / f"cpu{cpu}" / "cpuidle" / f"state{state}" / "disable"
                    path.parent.mkdir(parents=True)
                    path.write_text("0\n", encoding="utf-8")
                    if state > 0:
                        targets.append(path)
            evidence = Path(directory) / "cpuidle.json"

            def write_value(path: Path, value: int) -> str:
                path.write_text(f"{value}\n", encoding="utf-8")
                return str(value)

            failed = subprocess.CalledProcessError(7, ["diagnostic"])
            with patch(
                "benches.spark_cpuidle_guard._write_value", side_effect=write_value
            ), patch("benches.spark_cpuidle_guard.subprocess.run", side_effect=failed):
                with self.assertRaises(subprocess.CalledProcessError):
                    run_guard(
                        evidence=evidence,
                        cpus=(0, 19),
                        minimum_state=1,
                        command=["diagnostic"],
                        cpu_root=root,
                    )

            self.assertTrue(all(path.read_text().strip() == "0" for path in targets))
            report = json.loads(evidence.read_text(encoding="utf-8"))
            self.assertEqual(report["command_returncode"], 7)
            self.assertEqual(set(report["restored_verification"].values()), {0})
            self.assertEqual(len(report["writes"]), 8)

    def test_nonzero_initial_state_refuses_before_any_write(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "cpu"
            path = root / "cpu0" / "cpuidle" / "state1" / "disable"
            path.parent.mkdir(parents=True)
            path.write_text("1\n", encoding="utf-8")
            with patch("benches.spark_cpuidle_guard._write_value") as writer:
                with self.assertRaisesRegex(RuntimeError, "start enabled"):
                    run_guard(
                        evidence=Path(directory) / "cpuidle.json",
                        cpus=(0,),
                        minimum_state=1,
                        command=["diagnostic"],
                        cpu_root=root,
                    )
            writer.assert_not_called()


if __name__ == "__main__":
    unittest.main()

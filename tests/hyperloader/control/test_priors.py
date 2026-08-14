"""Calibration prior selection checks."""

from __future__ import annotations

import json
import unittest
from importlib.resources import files

from hyperloader.control import CpuCluster, MachineIdentity, spark_prior


def _profile() -> dict[str, object]:
    resource = files("hyperloader.control").joinpath("spark_prior.json")
    return json.loads(resource.read_text(encoding="utf-8"))


class SparkPriorTest(unittest.TestCase):
    """Prove the measured prior is narrow and provenance-labeled."""

    def test_named_machine_class_receives_its_campaign_record(self) -> None:
        profile = _profile()
        match = profile["match"]
        machine = MachineIdentity(
            " ".join(match["model_markers"]),
            (CpuCluster("all", (0,)),),
            1,
        )

        prior = spark_prior(machine)

        self.assertIsNotNone(prior)
        self.assertEqual(prior.machine, machine)
        self.assertEqual(
            prior.bandwidth_provenance,
            profile["calibration"]["bandwidth_provenance"],
        )
        self.assertIsNotNone(prior.staged_copy_tax)
        self.assertFalse(prior.staged_copy_tax.staging_is_profitable)

    def test_unrelated_machine_receives_no_spark_prior(self) -> None:
        machine = MachineIdentity("unrelated processor", (CpuCluster("all", (0,)),), 1)

        self.assertIsNone(spark_prior(machine))

    def test_topology_match_uses_the_shipped_profile(self) -> None:
        match = _profile()["match"]
        logical_cpus = int(match["logical_cpus"])
        machine = MachineIdentity(
            str(match["kernel_architecture"]),
            (
                CpuCluster(
                    "efficiency",
                    tuple(range(logical_cpus - 1)),
                    int(match["efficiency_frequency_hz"]),
                ),
                CpuCluster(
                    "performance",
                    (logical_cpus - 1,),
                    int(match["performance_frequency_hz"]),
                ),
            ),
            sum(int(value) for value in match["memory_range_bytes"]) // 2,
        )

        prior = spark_prior(machine)

        self.assertIsNotNone(prior)
        self.assertEqual(prior.machine, machine)


if __name__ == "__main__":
    unittest.main()

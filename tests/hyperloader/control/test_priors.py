"""Spark calibration prior selection checks."""

from __future__ import annotations

import unittest

from hyperloader.control import CpuCluster, MachineIdentity, spark_prior


class SparkPriorTest(unittest.TestCase):
    """Prove the measured prior is narrow and provenance-labeled."""

    def test_spark_class_receives_campaign_curves(self) -> None:
        machine = MachineIdentity(
            "NVIDIA Grace Cortex-X925 Cortex-A725",
            (
                CpuCluster("efficiency", tuple(range(10)), 2_808_000_000),
                CpuCluster("performance", tuple(range(10, 20)), 4_004_000_000),
            ),
            128 * 1024**3,
        )

        prior = spark_prior(machine)

        self.assertIsNotNone(prior)
        self.assertEqual(prior.machine, machine)
        self.assertEqual(prior.bandwidth_provenance, "derived-prior")
        self.assertEqual(prior.steal_curves[0].points[0].loss_fraction, 0.0204)
        self.assertEqual(prior.idle_state_tax.warm_duty_fraction, 0.05)
        self.assertEqual(prior.staged_copy_tax.batch_bytes, 262_144)

    def test_unrelated_machine_receives_no_spark_prior(self) -> None:
        machine = MachineIdentity("generic x86", (CpuCluster("all", (0,)),), 1024)

        self.assertIsNone(spark_prior(machine))


if __name__ == "__main__":
    unittest.main()

"""Machine identity and topology checks."""

from __future__ import annotations

import unittest

from hyperloader.control import CpuCluster, MachineIdentity, detect_machine_identity


class MachineIdentityTest(unittest.TestCase):
    """Prove stable keys and identity-sensitive invalidation inputs."""

    def test_key_is_stable_and_sensitive_to_every_hardware_term(self) -> None:
        base = MachineIdentity("cpu", (CpuCluster("all", (0, 1)),), 1024)
        same = MachineIdentity.from_dict(base.to_dict())
        changed_memory = MachineIdentity("cpu", base.clusters, 2048)
        changed_topology = MachineIdentity("cpu", (CpuCluster("all", (0,)),), 1024)

        self.assertEqual(base.cache_key, same.cache_key)
        self.assertNotEqual(base.cache_key, changed_memory.cache_key)
        self.assertNotEqual(base.cache_key, changed_topology.cache_key)

    def test_local_detection_returns_complete_identity(self) -> None:
        identity = detect_machine_identity()

        self.assertTrue(identity.cpu_model)
        self.assertGreater(identity.memory_bytes, 0)
        self.assertGreater(sum(len(cluster.logical_cpus) for cluster in identity.clusters), 0)


if __name__ == "__main__":
    unittest.main()

"""Host resource observation checks."""

from __future__ import annotations

import unittest

from hyperloader.process.resources import free_host_memory


class HostResourcesTest(unittest.TestCase):
    """Exercise the platform resource instrument used by sizing."""

    def test_available_host_memory_is_positive(self) -> None:
        self.assertGreater(free_host_memory(), 0)


if __name__ == "__main__":
    unittest.main()

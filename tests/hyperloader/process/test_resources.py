"""Host resource observation checks."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from hyperloader.process.resources import _macos_free_memory, free_host_memory


class HostResourcesTest(unittest.TestCase):
    """Exercise the platform resource instrument used by sizing."""

    def test_available_host_memory_is_positive(self) -> None:
        self.assertGreater(free_host_memory(), 0)

    def test_macos_routes_to_mach_host_statistics(self) -> None:
        with (
            patch("hyperloader.process.resources.sys.platform", "darwin"),
            patch(
                "hyperloader.process.resources._macos_free_memory",
                return_value=4096,
            ) as observe,
        ):
            self.assertEqual(free_host_memory(), 4096)
        observe.assert_called_once_with()

    def test_macos_counts_free_and_inactive_pages(self) -> None:
        library = _FakeMachLibrary(free_pages=3, inactive_pages=5)
        with (
            patch("hyperloader.process.resources.ctypes.CDLL", return_value=library),
            patch(
                "hyperloader.process.resources.os.sysconf",
                return_value=16_384,
                create=True,
            ),
        ):
            self.assertEqual(_macos_free_memory(), 8 * 16_384)


class _FakeFunction:
    def __init__(self, function):
        self.function = function
        self.argtypes = None
        self.restype = None

    def __call__(self, *arguments):
        return self.function(*arguments)


class _FakeMachLibrary:
    def __init__(self, *, free_pages: int, inactive_pages: int):
        self.mach_host_self = _FakeFunction(lambda: 7)
        self.host_statistics64 = _FakeFunction(
            lambda host, flavor, statistics, count: self._statistics(
                host,
                flavor,
                statistics,
                count,
                free_pages,
                inactive_pages,
            )
        )

    @staticmethod
    def _statistics(
        host,
        flavor,
        statistics,
        count,
        free_pages: int,
        inactive_pages: int,
    ) -> int:
        if host != 7 or flavor != 4 or count.value < 3:
            return 1
        statistics[0] = free_pages
        statistics[2] = inactive_pages
        return 0


if __name__ == "__main__":
    unittest.main()

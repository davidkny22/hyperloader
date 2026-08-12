"""Smoke tests for the installed package scaffold."""

import unittest

import hyperloader
from hyperloader import _hyperloader


class PackageSmokeTest(unittest.TestCase):
    """Exercise the public package import through the native extension."""

    def test_public_version_matches_distribution(self) -> None:
        self.assertEqual(hyperloader.__version__, "0.1.0")
        self.assertEqual(hyperloader.package_version(), "0.1.0")

    def test_native_build_reports_native_routing(self) -> None:
        self.assertIs(_hyperloader.IS_FALLBACK, False)


if __name__ == "__main__":
    unittest.main()

"""Smoke tests for the installed package scaffold."""

import unittest
from importlib.metadata import distributions
from pathlib import Path

import hyperloader
from hyperloader import _hyperloader


class PackageSmokeTest(unittest.TestCase):
    """Exercise the public package import through the native extension."""

    def test_public_version_matches_distribution(self) -> None:
        install_root = Path(hyperloader.__file__).resolve().parent.parent
        matches = [
            distribution
            for distribution in distributions(path=[str(install_root)])
            if distribution.metadata["Name"] == "hyperloader"
        ]
        self.assertEqual(len(matches), 1)
        installed = matches[0].version
        self.assertEqual(hyperloader.__version__, installed)
        self.assertEqual(hyperloader.package_version(), installed)

    def test_native_build_reports_native_routing(self) -> None:
        self.assertIs(_hyperloader.IS_FALLBACK, False)


if __name__ == "__main__":
    unittest.main()

"""Accelerator-interrupt route discovery tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hyperloader.control.interrupts import AcceleratorInterruptRoute


class AcceleratorInterruptRouteTest(unittest.TestCase):
    """Prove delta routing, steering refresh, and conservative fallback."""

    def test_refresh_uses_positive_deltas_instead_of_historical_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interrupts"
            _write_table(path, (100, 900, 0), (20, 0, 0))
            route = AcceleratorInterruptRoute.discover(path)
            self.assertIsNotNone(route)
            assert route is not None

            _write_table(path, (105, 900, 3), (22, 0, 0))
            self.assertEqual(route.refresh(), (0, 2))

            _write_table(path, (105, 907, 3), (22, 1, 0))
            self.assertEqual(route.refresh(), (1,))

    def test_unreadable_or_nonaccelerator_table_disables_route_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            self.assertIsNone(AcceleratorInterruptRoute.discover(missing))
            table = Path(directory) / "interrupts"
            table.write_text(
                "           CPU0 CPU1\n 10: 1 2 timer\n", encoding="utf-8"
            )
            self.assertIsNone(AcceleratorInterruptRoute.discover(table))


def _write_table(
    path: Path, first: tuple[int, int, int], second: tuple[int, int, int]
) -> None:
    path.write_text(
        "           CPU0 CPU1 CPU2\n"
        f" 484: {first[0]} {first[1]} {first[2]} PCI-MSI NVIDIA GPU\n"
        f" 486: {second[0]} {second[1]} {second[2]} PCI-MSI nvidia-modeset\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()

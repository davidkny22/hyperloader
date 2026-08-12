"""Installed Windows file-input routing assurance."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hyperloader import DataLoader, HyperConfig, _hyperloader
from hyperloader.config import IOConfig


class WindowsIOBackendTests(unittest.TestCase):
    """Exercise plan selection and exact IOCP completions through the wheel."""

    def test_public_loader_plan_selects_iocp(self) -> None:
        actual_selector = _hyperloader._io_backend_kind
        selected: list[str] = []

        def record(preference: str) -> str:
            selected.append(preference)
            return actual_selector(preference)

        with patch.object(_hyperloader, "_io_backend_kind", side_effect=record):
            loader = DataLoader(
                [1, 2], config=HyperConfig(io=IOConfig(backend="auto"))
            )
            loader.close()
        self.assertEqual(selected, ["auto"])
        self.assertEqual(actual_selector("auto"), "iocp")

    def test_installed_iocp_reads_exact_and_short_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "range.bin")
            path.write_bytes(b"0123456789abcdef")
            self.assertEqual(_hyperloader._read_range(path, 4, 6, "auto"), b"456789")
            self.assertEqual(_hyperloader._read_range(path, 15, 4, "iocp"), b"f")

    def test_explicit_linux_backend_is_rejected_on_windows(self) -> None:
        with self.assertRaisesRegex(ValueError, "unavailable on windows"):
            DataLoader([1], config=HyperConfig(io=IOConfig(backend="uring")))


if __name__ == "__main__":
    unittest.main()

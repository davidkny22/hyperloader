"""Native-free file-input refuge assurance."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hyperloader.fallback.native import io_backend_kind, read_range


class FallbackIOBackendTests(unittest.TestCase):
    """Keep positioned reads available without an extension binary."""

    def test_pread_refuge_reads_a_short_final_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "range.bin")
            path.write_bytes(b"abcdef")
            self.assertEqual(io_backend_kind("auto"), "pread")
            self.assertEqual(read_range(path, 4, 8), b"ef")

    def test_native_only_backends_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unavailable"):
            io_backend_kind("iocp")


if __name__ == "__main__":
    unittest.main()

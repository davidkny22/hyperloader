"""Installed macOS file, process, and named-arena routing assurance."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(sys.platform == "darwin", "requires macOS")
class MacOSPlatformTests(unittest.TestCase):
    """Exercise the Darwin-specific public execution route from an installed wheel."""

    def test_auto_io_uses_pread_and_reads_exact_ranges(self) -> None:
        from hyperloader import _hyperloader

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "range.bin")
            path.write_bytes(b"0123456789abcdef")
            self.assertEqual(_hyperloader._io_backend_kind("auto"), "pread")
            self.assertEqual(_hyperloader._read_range(path, 4, 6, "auto"), b"456789")
            self.assertEqual(_hyperloader._read_range(path, 15, 4, "pread"), b"f")

    def test_default_process_route_uses_spawn_and_cleans_up(self) -> None:
        from hyperloader import DataLoader, HyperConfig
        from hyperloader.config import IOConfig

        loader = DataLoader(
            range(8),
            batch_size=2,
            num_workers=2,
            seed=211,
            config=HyperConfig(io=IOConfig(backend="auto")),
        )
        pool = loader._process_pool
        self.assertIsNotNone(pool)
        assert pool is not None
        try:
            self.assertEqual(
                pool._worker_set._context.get_start_method(),
                "spawn",
            )
            self.assertEqual(
                [batch.tolist() for batch in loader],
                [[0, 1], [2, 3], [4, 5], [6, 7]],
            )
        finally:
            loader.close()
        self.assertEqual(pool.worker_pids, ())


if __name__ == "__main__":
    unittest.main()

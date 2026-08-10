"""Installed public checks for worker and named-region hygiene."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from hyperloader import DataLoader
from hyperloader.process import ProcessPool


class InterruptDataset:
    """Raise the process-wide interruption type through worker delivery."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _index: int) -> int:
        raise KeyboardInterrupt("synthetic interrupt")


def registry_path(cache: str | Path) -> Path:
    """Resolve the isolated Windows registry used by the installed artifact."""
    return Path(cache) / "hyperloader" / "regions.jsonl"


def registry_rows(path: Path) -> list[str]:
    """Return complete nonempty ownership records."""
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def process_has_exited(pid: int) -> bool:
    """Observe process exit without taking ownership of the process."""
    if os.name == "nt":
        synchronize = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, pid)
        if handle == 0:
            return True
        try:
            return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == 0
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    if sys.platform.startswith("linux"):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(
                ") ", 1
            )[1]
        except (FileNotFoundError, ProcessLookupError):
            return True
        if fields.startswith("Z "):
            return True
    return False


def wait_for_exit(pids: tuple[int, ...] | list[int], timeout: float = 5.0) -> None:
    """Require every observed worker identity to exit within the cleanup bound."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(process_has_exited(pid) for pid in pids):
            return
        time.sleep(0.02)
    live = [pid for pid in pids if not process_has_exited(pid)]
    raise AssertionError(f"worker processes remained live after cleanup: {live}")


class SharedMemoryHygieneGate(unittest.TestCase):
    """Prove normal, interrupted, and crash cleanup through DataLoader."""

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux procfs only")
    def test_zombie_worker_is_observed_as_exited(self) -> None:
        with (
            mock.patch.object(os, "kill"),
            mock.patch.object(
                Path,
                "read_text",
                return_value="130 (python worker) Z 1 2 3",
            ),
        ):
            self.assertTrue(process_has_exited(130))

    def test_explicit_close_releases_workers_and_registry(self) -> None:
        with TemporaryDirectory() as cache, mock.patch.dict(
            os.environ, {"HYPERLOADER_CACHE_HOME": cache}
        ):
            loader = DataLoader(range(4), batch_size=1, num_workers=2, seed=73)
            pool = loader._process_pool
            pids = pool.worker_pids
            path = registry_path(cache)
            self.assertGreater(len(registry_rows(path)), 0)
            mutation = (
                mock.patch.object(ProcessPool, "_release_handles", lambda _self: None)
                if os.environ.get("HYPERLOADER_SHM_HYGIENE_MUTATION")
                == "retain-native-owner"
                else nullcontext()
            )
            try:
                with mutation:
                    loader.close()
                wait_for_exit(pids)
                self.assertEqual(registry_rows(path), [])
            finally:
                if pool._resources is not None:
                    pool._release_handles()

    def test_keyboard_interrupt_closes_loader(self) -> None:
        with TemporaryDirectory() as cache, mock.patch.dict(
            os.environ, {"HYPERLOADER_CACHE_HOME": cache}
        ):
            loader = DataLoader(
                InterruptDataset(), batch_size=1, num_workers=1, seed=79
            )
            pids = loader._process_pool.worker_pids
            with self.assertRaisesRegex(KeyboardInterrupt, "synthetic interrupt"):
                next(iter(loader))
            self.assertIsNone(loader._process_pool)
            wait_for_exit(pids)
            self.assertEqual(registry_rows(registry_path(cache)), [])

    def test_parent_crash_kills_blocked_workers_and_next_construct_reaps(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(__file__).parents[2]
            cache = Path(directory) / "cache"
            state_path = Path(directory) / "state.json"
            environment = os.environ.copy()
            environment["HYPERLOADER_CACHE_HOME"] = str(cache)
            child = subprocess.run(
                [
                    sys.executable,
                    str(root / "tests" / "hyperloader" / "shm_hygiene_child.py"),
                    "--state",
                    str(state_path),
                ],
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(child.returncode, 23, child.stdout + child.stderr)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            workers = [int(pid) for pid in state["workers"]]
            wait_for_exit(workers)
            path = Path(state["registry"])
            self.assertGreater(len(registry_rows(path)), 0)

            with mock.patch.dict(
                os.environ, {"HYPERLOADER_CACHE_HOME": str(cache)}
            ):
                replacement = DataLoader(
                    range(1), batch_size=1, num_workers=1, seed=83
                )
                replacement.close()
            self.assertEqual(registry_rows(path), [])


if __name__ == "__main__":
    unittest.main()

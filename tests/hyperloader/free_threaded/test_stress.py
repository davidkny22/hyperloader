"""Installed free-threaded race, lifecycle, and restoration assurance."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from hyperloader import DataLoader, rng
from hyperloader.thread import ThreadPool
from hyperloader.thread.gil import free_threaded_build, gil_enabled
from hyperloader.verify import _bit_equal


class ConcurrentDataset:
    """Produce coordinate-bound values while exposing simultaneous calls."""

    def __init__(self) -> None:
        self._active = 0
        self._lock = threading.Lock()
        self.max_active = 0
        self.thread_ids: set[int] = set()

    def __len__(self) -> int:
        return 24

    def __getitem__(self, index: int) -> dict[str, object]:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            self.thread_ids.add(threading.get_ident())
        try:
            time.sleep(0.0005)
            return {
                "index": index,
                "numpy": np.asarray(rng("numpy").random(2), dtype=np.float64),
                "python": rng("random").getrandbits(31),
                "torch": torch.rand(2, generator=rng()),
            }
        finally:
            with self._lock:
                self._active -= 1


class FreeThreadedStressGate(unittest.TestCase):
    """Exercise sustained declared-thread execution on exact FT wheels."""

    def test_parallel_loaders_preserve_coordinate_streams(self) -> None:
        self.assertTrue(free_threaded_build())
        self.assertIs(gil_enabled(), False)
        baseline, _, _ = _collect(211)
        original = ThreadPool.submit

        def shifted_submit(
            pool: ThreadPool,
            epoch: int,
            position: int,
            index: int,
            coordinate: int | None = None,
        ):
            resolved = position if coordinate is None else coordinate
            return original(pool, epoch, position, index, resolved + 1)

        route = (
            mock.patch.object(ThreadPool, "submit", shifted_submit)
            if os.environ.get("HYPERLOADER_FT_STRESS_MUTATE") == "1"
            else mock.patch.object(ThreadPool, "submit", original)
        )
        with route, ThreadPoolExecutor(max_workers=4) as callers:
            futures = [callers.submit(_collect, 211) for _ in range(8)]
        for future in futures:
            values, max_active, thread_count = future.result()
            self.assertTrue(_bit_equal(values, baseline))
            self.assertGreaterEqual(max_active, 2)
            self.assertGreaterEqual(thread_count, 2)

    def test_repeated_early_close_releases_every_worker_thread(self) -> None:
        baseline = _hyperloader_thread_count()

        def churn(seed: int) -> None:
            for cycle in range(6):
                loader = DataLoader(
                    ConcurrentDataset(),
                    batch_size=3,
                    num_workers=4,
                    seed=seed + cycle,
                    thread_safe=True,
                )
                iterator = iter(loader)
                next(iterator)
                loader.close()
                loader.close()

        with ThreadPoolExecutor(max_workers=6) as callers:
            futures = [callers.submit(churn, 300 + index * 20) for index in range(6)]
        for future in futures:
            future.result()
        deadline = time.monotonic() + 3.0
        while _hyperloader_thread_count() > baseline and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(_hyperloader_thread_count(), baseline)

    def test_non_declaring_extension_restoration_is_reported_once(self) -> None:
        fixture = Path(__file__).with_name("fixtures") / "gil_restoring.c"
        runner = Path(__file__).with_name("restoration_runner.py")
        with tempfile.TemporaryDirectory() as temporary:
            build_root = Path(temporary)
            command = [
                sys.executable,
                "-c",
                (
                    "from setuptools import Extension, setup; "
                    f"setup(name='gil-restoring', ext_modules=[Extension("
                    f"'gil_restoring', [r'{fixture}'])])"
                ),
                "build_ext",
                "--build-lib",
                str(build_root),
            ]
            built = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=120
            )
            if built.returncode != 0:
                self.fail(f"extension build failed\n{built.stdout}\n{built.stderr}")
            restored = subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    str(build_root),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if restored.returncode != 0:
                self.fail(
                    f"restoration subprocess failed\n{restored.stdout}\n"
                    f"{restored.stderr}"
                )
            record = json.loads(restored.stdout.strip())
            self.assertEqual(record, {"events": 1, "values": [0, 1, 4, 9]})

    def test_standard_build_identity_is_not_an_ft_stress_target(self) -> None:
        with mock.patch(
            "hyperloader.thread.gil.sysconfig.get_config_var", return_value=0
        ):
            self.assertFalse(free_threaded_build())
        self.assertEqual(sysconfig.get_config_var("Py_GIL_DISABLED"), 1)


def _collect(seed: int) -> tuple[list[object], int, int]:
    dataset = ConcurrentDataset()
    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=True,
        num_workers=4,
        seed=seed,
        thread_safe=True,
    )
    try:
        values = list(loader)
        return values, dataset.max_active, len(dataset.thread_ids)
    finally:
        loader.close()


def _hyperloader_thread_count() -> int:
    return sum(thread.name.startswith("hyperloader") for thread in threading.enumerate())


if __name__ == "__main__":
    unittest.main()

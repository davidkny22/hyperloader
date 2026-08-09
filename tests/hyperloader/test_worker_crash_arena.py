"""Installed public gate for worker death, arena reclamation, and recovery."""

from __future__ import annotations

import os
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

from hyperloader import DataLoader, HyperConfig
from hyperloader.config import ExecutorConfig
from hyperloader.process import ProcessPool


class CrashOnceDataset:
    """Exit one routed worker after persisting a cross-process sentinel."""

    def __init__(self, sentinel: str) -> None:
        self.sentinel = sentinel

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> int:
        path = Path(self.sentinel)
        if index == 1 and not path.exists():
            path.write_text("crashed", encoding="utf-8")
            os._exit(17)
        return index


class HealthySkewDataset:
    """Delay one sample without ending its worker process."""

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> int:
        if index == 1:
            time.sleep(0.1)
        return index


def record_worker_init(worker: int) -> None:
    """Record each process that executes worker initialization."""
    directory = Path(os.environ["HYPERLOADER_RECOVERY_INIT_LOG"])
    (directory / f"worker-{worker}-{os.getpid()}.log").write_text(
        "initialized\n", encoding="utf-8"
    )


def _reclaim_without_restart(pool: ProcessPool, worker: int) -> list[int]:
    """Plant the omission of replacement-worker creation for mutation proof."""
    return sorted(pool._resources.reclaim_dead_worker(worker))


class WorkerCrashArenaGate(unittest.TestCase):
    """Prove death-only reclamation, surfaced context, and configured restart."""

    def test_restart_reclaims_and_replays_dead_worker_positions(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            sentinel = str(Path(directory) / "worker-crashed.txt")
            previous_log = os.environ.get("HYPERLOADER_RECOVERY_INIT_LOG")
            os.environ["HYPERLOADER_RECOVERY_INIT_LOG"] = directory
            config = HyperConfig(
                executor=ExecutorConfig(process_ceiling=2, on_worker_death="restart")
            )
            loader = DataLoader(
                CrashOnceDataset(sentinel),
                batch_size=1,
                num_workers=2,
                seed=59,
                config=config,
                worker_init_fn=record_worker_init,
            )
            iterator = iter(loader)
            original_pids = loader._process_pool.worker_pids
            mutation = (
                mock.patch.object(ProcessPool, "_restart_worker", _reclaim_without_restart)
                if os.environ.get("HYPERLOADER_WORKER_CRASH_MUTATION")
                == "omit-restart"
                else nullcontext()
            )
            try:
                self.assertEqual(int(next(iterator).item()), 0)
                with mutation, self.assertRaisesRegex(
                    RuntimeError, r"worker 1 exited.*positions \[1\]"
                ):
                    next(iterator)
                replacement_pids = loader._process_pool.worker_pids
                self.assertEqual(replacement_pids[0], original_pids[0])
                self.assertNotEqual(replacement_pids[1], original_pids[1])
                delivered = [int(next(iterator).item()) for _ in range(3)]
            finally:
                loader.close()
                if previous_log is None:
                    os.environ.pop("HYPERLOADER_RECOVERY_INIT_LOG", None)
                else:
                    os.environ["HYPERLOADER_RECOVERY_INIT_LOG"] = previous_log

            init_logs = [path.name for path in Path(directory).glob("worker-*.log")]

        self.assertEqual(delivered, [1, 2, 3])
        self.assertEqual(sum(name.startswith("worker-0-") for name in init_logs), 1)
        self.assertEqual(sum(name.startswith("worker-1-") for name in init_logs), 2)

    def test_close_policy_reclaims_without_replacement(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            loader = DataLoader(
                CrashOnceDataset(str(Path(directory) / "close-crashed.txt")),
                batch_size=1,
                num_workers=2,
                seed=61,
            )
            iterator = iter(loader)
            try:
                self.assertEqual(int(next(iterator).item()), 0)
                with self.assertRaisesRegex(
                    RuntimeError, r"closed after reclaiming positions \[1\]"
                ):
                    next(iterator)
                self.assertIsNone(loader._process_pool)
            finally:
                loader.close()

    def test_healthy_slow_worker_is_never_reclaimed(self) -> None:
        config = HyperConfig(
            executor=ExecutorConfig(process_ceiling=2, on_worker_death="restart")
        )
        loader = DataLoader(
            HealthySkewDataset(), batch_size=1, num_workers=2, seed=67, config=config
        )
        original_pids = loader._process_pool.worker_pids
        try:
            delivered = [int(batch.item()) for batch in loader]
            final_pids = loader._process_pool.worker_pids
        finally:
            loader.close()

        self.assertEqual(delivered, [0, 1, 2])
        self.assertEqual(final_pids, original_pids)


if __name__ == "__main__":
    unittest.main()

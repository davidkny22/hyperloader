"""Torch-shaped process exceptions, timeout, and iterator lifecycle tests."""

from __future__ import annotations

import time
import unittest
from pathlib import Path

from hyperloader import DataLoader
from hyperloader.process import ProcessPool


class FailingDataset:
    """Raise a selected built-in exception away from the probe position."""

    def __init__(self, exception_type: type[BaseException]) -> None:
        self.exception_type = exception_type

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> int:
        if index == 1:
            raise self.exception_type("sample failed")
        return index


class MultiArgumentError(Exception):
    """Require reconstruction arguments unavailable across the worker boundary."""

    def __init__(self, code: int, detail: str) -> None:
        super().__init__(code, detail)


class MultiArgumentDataset:
    """Raise an exception that cannot be reconstructed from one message."""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> int:
        if index == 1:
            raise MultiArgumentError(7, "sample failed")
        return index


class OrderedFailureDataset:
    """Discover a later exception before an earlier slow position completes."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> int:
        if index == 1:
            time.sleep(0.15)
        if index == 2:
            raise ValueError("ordered failure")
        return index


class TimeoutDataset:
    """Exceed the consumer timeout away from the construction probe."""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> int:
        if index == 1:
            time.sleep(10)
        return index


class FailOnceDataset:
    """Use an external sentinel so a fresh iterator can recover."""

    def __init__(self, sentinel: str) -> None:
        self.sentinel = sentinel

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> int:
        path = Path(self.sentinel)
        if index == 1 and not path.exists():
            path.write_text("failed", encoding="utf-8")
            raise ValueError("fail once")
        return index


def fail_worker_init(_worker_id: int) -> None:
    """Raise before fetcher use so delivery must defer the startup error."""
    raise ValueError("initialization failed")


class ProcessExceptionTest(unittest.TestCase):
    """Exercise exception reconstruction and consumer-visible lifecycle rules."""

    def test_builtin_types_keep_torch_message_shape(self) -> None:
        for exception_type in (ValueError, TypeError, KeyError):
            with self.subTest(exception_type=exception_type):
                pool = ProcessPool(FailingDataset(exception_type), 1, 29, 0, 0, 0)
                try:
                    with self.assertRaises(exception_type) as raised:
                        pool.execute(0, 1, 1)
                finally:
                    pool.close()

                message = str(raised.exception)
                self.assertIn(
                    f"Caught {exception_type.__name__} in DataLoader worker process 0.",
                    message,
                )
                self.assertIn("Original Traceback", message)
                self.assertIn("sample failed", message)

    def test_unconstructible_type_falls_back_to_runtime_error(self) -> None:
        pool = ProcessPool(MultiArgumentDataset(), 1, 31, 0, 0, 0)
        try:
            with self.assertRaisesRegex(RuntimeError, "Caught MultiArgumentError"):
                pool.execute(0, 1, 1)
        finally:
            pool.close()

    def test_worker_init_failure_surfaces_on_first_next(self) -> None:
        loader = DataLoader(
            [0], num_workers=1, seed=37, worker_init_fn=fail_worker_init
        )
        iterator = iter(loader)
        try:
            with self.assertRaisesRegex(ValueError, "initialization failed"):
                next(iterator)
            with self.assertRaisesRegex(RuntimeError, "no longer active"):
                next(iterator)
        finally:
            loader.close()

    def test_exception_waits_for_its_delivery_position(self) -> None:
        loader = DataLoader(
            OrderedFailureDataset(), batch_size=1, num_workers=2, seed=41
        )
        iterator = iter(loader)
        try:
            self.assertEqual(int(next(iterator).item()), 0)
            self.assertEqual(int(next(iterator).item()), 1)
            with self.assertRaisesRegex(ValueError, "ordered failure"):
                next(iterator)
        finally:
            loader.close()

    def test_timeout_kills_iterator_and_pool(self) -> None:
        loader = DataLoader(
            TimeoutDataset(), batch_size=1, num_workers=1, seed=43, timeout=0.05
        )
        iterator = iter(loader)
        try:
            self.assertEqual(int(next(iterator).item()), 0)
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "timed out after 0.05 seconds"):
                next(iterator)
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertIsNone(loader._process_pool)
            with self.assertRaisesRegex(RuntimeError, "no longer active"):
                next(iterator)
        finally:
            loader.close()

    def test_fresh_iterator_recovers_after_user_exception(self) -> None:
        with self.subTest("external failure clears after first raise"):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as directory:
                sentinel = str(Path(directory) / "failed.txt")
                loader = DataLoader(
                    FailOnceDataset(sentinel), batch_size=1, num_workers=2, seed=47
                )
                failed = iter(loader)
                try:
                    self.assertEqual(int(next(failed).item()), 0)
                    with self.assertRaisesRegex(ValueError, "fail once"):
                        next(failed)
                    with self.assertWarnsRegex(UserWarning, "advanced the epoch"):
                        replayed = [int(batch.item()) for batch in loader]
                finally:
                    loader.close()

        self.assertEqual(replayed, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()

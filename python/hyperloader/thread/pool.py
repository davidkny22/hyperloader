"""Persistent workers for declared thread-safe user code."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from hyperloader import _hyperloader
from hyperloader.rng import _user_code_context

from .gil import GilRestorationDetector


class ThreadPool:
    """Execute user dataset calls in one shared-address-space worker set."""

    def __init__(
        self,
        dataset: Any,
        worker_count: int,
        root_seed: int,
        worker_init_fn: Any,
        telemetry: Any | None,
    ) -> None:
        self._dataset = dataset
        self._worker_count = worker_count
        self._root_seed = root_seed
        self._worker_init_fn = worker_init_fn
        self._next_worker_id = 0
        self._worker_id_lock = threading.Lock()
        self._local = threading.local()
        self._gil = GilRestorationDetector(telemetry)
        self._executor = ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="hyperloader"
        )
        self._closed = False

    @property
    def worker_count(self) -> int:
        """Return the fixed number of shared-address-space workers."""
        return self._worker_count

    def submit(
        self, epoch: int, position: int, index: int, coordinate: int | None = None
    ) -> Future[tuple[Any, int]]:
        """Submit one coordinate-bound dataset call."""
        if self._closed:
            raise RuntimeError("thread pool is closed")
        return self._executor.submit(
            self._evaluate,
            epoch,
            position if coordinate is None else coordinate,
            index,
        )

    def _evaluate(self, epoch: int, position: int, index: int) -> tuple[Any, int]:
        self._initialize_worker()
        sample = _hyperloader._sample_rng_context(self._root_seed, epoch, position)
        started = time.perf_counter_ns()
        try:
            with _user_code_context(sample):
                value = self._dataset[index]
        finally:
            self._gil.observe()
        return value, max(1, time.perf_counter_ns() - started)

    def _initialize_worker(self) -> None:
        if getattr(self._local, "initialized", False):
            return
        with self._worker_id_lock:
            worker_id = self._next_worker_id
            self._next_worker_id += 1
        if self._worker_init_fn is not None:
            self._worker_init_fn(worker_id)
        self._local.initialized = True

    def close(self) -> None:
        """Cancel queued calls and join active worker threads."""
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __del__(self) -> None:
        self.close()

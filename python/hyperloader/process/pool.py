"""Persistent process ownership and black-box request lifecycle."""

from __future__ import annotations

import importlib
import multiprocessing as mp
import pickle
import sys
import time
from collections import deque
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from hyperloader import _hyperloader

from .worker import BLACK_BOX_STAGE, worker_main

POLL_SECONDS = 0.005
SHUTDOWN_SECONDS = 5.0


class ProcessPool:
    """One persistent spawn-safe worker set over native arena transport."""

    def __init__(
        self,
        dataset: Any,
        worker_count: int,
        root_seed: int,
        probe_epoch: int,
        probe_position: int,
        probe_index: int,
        *,
        worker_init_fn: Any = None,
        multiprocessing_context: Any = None,
        timeout: float = 0,
        registry_path: str | Path | None = None,
        queue_capacity: int = 2,
    ) -> None:
        self._context = resolve_context(multiprocessing_context)
        self._timeout = timeout
        self._closed = False
        self._next_worker = 0
        self._resources: Any = None
        self._controls: list[Connection] = []
        self._workers: list[mp.Process] = []
        self._probe_key = (probe_epoch, probe_position, probe_index)
        self._probe_payload: bytes | None = None
        self._immediate: deque[tuple[int, int, int, bytes]] = deque()
        dataset_payload = pickle.dumps((dataset, worker_init_fn), protocol=5)
        try:
            for worker_id in range(worker_count):
                owner, child = self._context.Pipe(duplex=True)
                probe = self._probe_key if worker_id == 0 else None
                process = self._context.Process(
                    target=worker_main,
                    args=(
                        child,
                        dataset_payload,
                        worker_id,
                        worker_count,
                        root_seed,
                        probe,
                    ),
                    daemon=True,
                )
                process.start()
                child.close()
                self._controls.append(owner)
                self._workers.append(process)
            status, probe_payload = self._receive_probe()
            if status != 0:
                reraise_worker_exception(probe_payload, 0)
            payload_capacity = max(1, len(probe_payload))
            exception_capacity = max(65_536, payload_capacity * 2)
            self._resources = _hyperloader._ProcessResources(
                worker_count,
                queue_capacity,
                payload_capacity,
                exception_capacity,
                None if registry_path is None else Path(registry_path),
            )
            for worker_id, control in enumerate(self._controls):
                control.send(("attach", self._resources.descriptor(worker_id)))
            self._probe_payload = probe_payload
        except BaseException:
            self.close()
            raise

    @property
    def worker_pids(self) -> tuple[int, ...]:
        """Return stable process identifiers for liveness tests and diagnosis."""
        return tuple(process.pid for process in self._workers if process.pid is not None)

    @property
    def worker_count(self) -> int:
        """Return the fixed number of persistent process workers."""
        return len(self._workers)

    def try_submit(
        self, epoch: int, position: int, index: int, worker: int
    ) -> bool:
        """Attempt one targeted black-box dispatch without waiting for capacity."""
        if self._closed:
            raise RuntimeError("process pool is closed")
        key = (epoch, position, index)
        if self._probe_payload is not None and key == self._probe_key:
            if worker != 0:
                raise RuntimeError("construction probe must retain worker zero routing")
            self._immediate.append((worker, position, 0, self._probe_payload))
            self._probe_payload = None
            return True
        return self._resources.try_submit(
            epoch, position, index, BLACK_BOX_STAGE, worker
        )

    def try_receive(self, worker: int) -> tuple[int, int, bytes] | None:
        """Attempt one completion without imposing consumer delivery order."""
        if self._immediate and self._immediate[0][0] == worker:
            _, position, status, payload = self._immediate.popleft()
            return position, status, payload
        return self._resources.try_receive(worker)

    def decode(self, status: int, payload: bytes, worker: int) -> Any:
        """Reconstruct a successful sample or re-raise its worker exception."""
        if status == 0:
            return pickle.loads(payload)
        reraise_worker_exception(payload, worker)

    def execute(self, epoch: int, position: int, index: int) -> Any:
        """Execute one black-box sample and reconstruct its result or exception."""
        if self._closed:
            raise RuntimeError("process pool is closed")
        worker = self._next_worker
        self._next_worker = (self._next_worker + 1) % len(self._workers)
        deadline = None if self._timeout == 0 else time.monotonic() + self._timeout
        while not self.try_submit(epoch, position, index, worker):
            self._check_worker(worker, deadline)
            time.sleep(POLL_SECONDS)
        while True:
            completion = self.try_receive(worker)
            if completion is not None:
                received_position, status, payload = completion
                if received_position != position:
                    raise RuntimeError("process completion position does not match dispatch")
                return self.decode(status, payload, worker)
            self._check_worker(worker, deadline)
            time.sleep(POLL_SECONDS)

    def deadline(self) -> float | None:
        """Create the current request's timeout deadline."""
        return None if self._timeout == 0 else time.monotonic() + self._timeout

    def check_workers(self, deadline: float | None) -> None:
        """Validate every process and the consumer timeout deadline."""
        for worker in range(len(self._workers)):
            self._check_worker(worker, deadline)

    def close(self) -> None:
        """Stop workers, reclaim native resources, and make closure idempotent."""
        if self._closed:
            return
        self._closed = True
        for control in self._controls:
            try:
                control.send(("stop",))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self._workers:
            process.join(SHUTDOWN_SECONDS)
            if process.is_alive():
                process.terminate()
                process.join(SHUTDOWN_SECONDS)
        for control in self._controls:
            control.close()
        self._controls.clear()
        self._workers.clear()
        self._resources = None

    def _receive_probe(self) -> tuple[int, bytes]:
        deadline = None if self._timeout == 0 else time.monotonic() + self._timeout
        while True:
            if self._controls[0].poll(POLL_SECONDS):
                kind, status, payload = self._controls[0].recv()
                if kind != "probe":
                    raise RuntimeError("worker returned an invalid probe response")
                return status, payload
            self._check_worker(0, deadline)

    def _check_worker(self, worker: int, deadline: float | None) -> None:
        process = self._workers[worker]
        if not process.is_alive():
            if self._resources is not None:
                self._resources.reclaim_dead_worker(worker)
            raise RuntimeError(
                f"hyperloader worker {worker} exited with code {process.exitcode}"
            )
        if deadline is not None and time.monotonic() >= deadline:
            raise RuntimeError(f"DataLoader timed out after {self._timeout} seconds")

    def __del__(self) -> None:
        self.close()


def resolve_context(requested: Any) -> Any:
    """Resolve an explicit context or the spawn-safe platform default."""
    if requested is not None:
        return mp.get_context(requested) if isinstance(requested, str) else requested
    if sys.platform in {"win32", "darwin"}:
        return mp.get_context("spawn")
    return mp.get_context("forkserver")


def reraise_worker_exception(payload: bytes, worker: int) -> None:
    """Reconstruct torch-shaped exception context without retaining worker frames."""
    module_name, qualname, original_message, formatted = pickle.loads(payload)
    message = (
        f"Caught {qualname} in hyperloader worker process {worker}.\n"
        f"Original traceback:\n{formatted}"
    )
    try:
        exception_type: Any = importlib.import_module(module_name)
        for component in qualname.split("."):
            exception_type = getattr(exception_type, component)
        if not isinstance(exception_type, type) or not issubclass(
            exception_type, BaseException
        ):
            raise TypeError
        raise exception_type(message)
    except (AttributeError, ImportError, TypeError):
        raise RuntimeError(f"{message}\nOriginal message: {original_message}") from None

"""Persistent process ownership and black-box request lifecycle."""

from __future__ import annotations

import multiprocessing as mp
import pickle
import sys
import time
from collections import deque
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from hyperloader import _hyperloader

from .exceptions import WorkerDied, reraise_worker_exception
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
        on_worker_death: str = "close",
    ) -> None:
        self._context = resolve_context(multiprocessing_context)
        self._timeout = timeout
        self._closed = False
        self._next_worker = 0
        self._on_worker_death = on_worker_death
        self._root_seed = root_seed
        self._worker_total = worker_count
        self._resources: Any = None
        self._controls: list[Connection] = []
        self._workers: list[mp.Process] = []
        self._probe_key = (probe_epoch, probe_position, probe_index)
        self._probe_status = 0
        self._probe_payload: bytes | None = None
        self._immediate: deque[tuple[int, int, int, bytes]] = deque()
        self._pending: dict[tuple[int, int], tuple[int, int]] = {}
        self._dataset_payload = pickle.dumps((dataset, worker_init_fn), protocol=5)
        try:
            for worker_id in range(worker_count):
                probe = self._probe_key if worker_id == 0 else None
                owner, process = self._launch_worker(worker_id, probe)
                self._controls.append(owner)
                self._workers.append(process)
            status, probe_payload = self._receive_probe()
            self._probe_status = status
            payload_capacity = 262_144 if status != 0 else max(1, len(probe_payload))
            exception_capacity = max(65_536, len(probe_payload), payload_capacity * 2)
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
            self._immediate.append(
                (worker, position, self._probe_status, self._probe_payload)
            )
            self._probe_payload = None
            return True
        accepted = self._resources.try_submit(
            epoch, position, index, BLACK_BOX_STAGE, worker
        )
        if accepted:
            self._pending[(worker, position)] = (epoch, index)
        return accepted

    def try_receive(self, worker: int) -> tuple[int, int, bytes] | None:
        """Attempt one completion without imposing consumer delivery order."""
        if self._immediate and self._immediate[0][0] == worker:
            _, position, status, payload = self._immediate.popleft()
            return position, status, payload
        completion = self._resources.try_receive(worker)
        if completion is not None:
            self._pending.pop((worker, completion[0]), None)
        return completion

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
        self._release_handles()

    def abort(self) -> None:
        """Terminate a timed-out pool immediately, then release native handles."""
        if self._closed:
            return
        self._closed = True
        for process in self._workers:
            if process.is_alive():
                process.terminate()
        for process in self._workers:
            process.join(SHUTDOWN_SECONDS)
        self._release_handles()

    def _release_handles(self) -> None:
        """Close controls and drop native owners after every shutdown mode."""
        for control in self._controls:
            control.close()
        self._controls.clear()
        self._workers.clear()
        self._pending.clear()
        self._resources = None

    def _receive_probe(self) -> tuple[int, bytes]:
        while True:
            if self._controls[0].poll(POLL_SECONDS):
                kind, status, payload = self._controls[0].recv()
                if kind != "probe":
                    raise RuntimeError("worker returned an invalid probe response")
                return status, payload
            self._check_worker(0, None)

    def _launch_worker(
        self, worker: int, probe: tuple[int, int, int] | None
    ) -> tuple[Connection, mp.Process]:
        owner, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=worker_main,
            args=(
                child,
                self._dataset_payload,
                worker,
                self._worker_total,
                self._root_seed,
                probe,
            ),
            daemon=True,
        )
        process.start()
        child.close()
        return owner, process

    def _restart_worker(self, worker: int) -> list[int]:
        positions = sorted(self._resources.restart_worker(worker))
        self._workers[worker].join(0)
        self._controls[worker].close()
        owner, process = self._launch_worker(worker, None)
        self._controls[worker] = owner
        self._workers[worker] = process
        owner.send(("attach", self._resources.descriptor(worker)))
        for position in positions:
            epoch, index = self._pending[(worker, position)]
            if not self._resources.try_submit(
                epoch, position, index, BLACK_BOX_STAGE, worker
            ):
                raise RuntimeError("replacement worker transport rejected recovered work")
        return positions

    def _check_worker(self, worker: int, deadline: float | None) -> None:
        process = self._workers[worker]
        if not process.is_alive():
            exitcode = process.exitcode
            positions: list[int] = []
            if self._resources is not None:
                if self._on_worker_death == "restart":
                    positions = self._restart_worker(worker)
                    raise WorkerDied(
                        f"hyperloader worker {worker} exited with code {exitcode}; "
                        f"restarted after reclaiming positions {positions}"
                    )
                positions = self._resources.reclaim_dead_worker(worker)
            self.abort()
            raise RuntimeError(
                f"hyperloader worker {worker} exited with code {exitcode}; "
                f"closed after reclaiming positions {positions}"
            )
        if deadline is not None and time.monotonic() >= deadline:
            self.abort()
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

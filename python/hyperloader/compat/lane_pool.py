"""Native command, arena, and process ownership for Torch-compatible lanes."""

from __future__ import annotations

import multiprocessing as mp
import pickle
import sys
import time
from multiprocessing.connection import Connection, wait
from typing import Any

from hyperloader import _hyperloader
from hyperloader.process.exceptions import reraise_worker_exception
from hyperloader.process.serialization import ResultDecoder, encode_multiprocessing

from .lane_worker import COMPAT_STAGE, lane_worker_main

POLL_SECONDS = 0.005
PAYLOAD_CAPACITY = 1_048_576
EXCEPTION_CAPACITY = 65_536


class CompatLanePool:
    """Own one native transport and one operating-system process per Torch lane."""

    def __init__(
        self,
        loader: Any,
        base_seed: int,
        lane_states: dict[int, bytes],
        *,
        capture_state: bool,
    ) -> None:
        self._timeout = loader.timeout
        self._closed = False
        self._decoder = ResultDecoder()
        self._controls: list[Connection] = []
        self._processes: list[mp.Process] = []
        self._pending: dict[int, int] = {}
        prefetch = loader._compat_reference.prefetch_factor or 2
        capacity = max(2, 1 << max(1, prefetch).bit_length())
        self._resources = _hyperloader._ProcessResources(
            loader.num_workers,
            capacity,
            PAYLOAD_CAPACITY,
            EXCEPTION_CAPACITY,
            None,
        )
        reference = loader._compat_reference
        payload = encode_multiprocessing(
            (
                loader.dataset,
                reference.collate_fn,
                loader.worker_init_fn,
                reference._auto_collation,
            )
        )
        context = _resolve_context(loader.multiprocessing_context)
        try:
            for worker in range(loader.num_workers):
                owner, child = context.Pipe(duplex=True)
                process = context.Process(
                    target=lane_worker_main,
                    args=(
                        child,
                        payload,
                        worker,
                        loader.num_workers,
                        base_seed,
                        capture_state,
                        lane_states.get(worker),
                    ),
                    daemon=True,
                )
                process.start()
                child.close()
                self._controls.append(owner)
                self._processes.append(process)
            for worker, control in enumerate(self._controls):
                control.send(("attach", self._resources.descriptor(worker)))
        except BaseException:
            self.close()
            raise

    @property
    def worker_pids(self) -> tuple[int, ...]:
        """Return live process identifiers in stable lane order."""
        return tuple(
            process.pid for process in self._processes if process.pid is not None
        )

    @property
    def has_pending(self) -> bool:
        """Report whether any accepted command remains incomplete."""
        return bool(self._pending)

    @property
    def pending_workers(self) -> frozenset[int]:
        """Return lanes that own at least one incomplete command."""
        return frozenset(self._pending.values())

    @property
    def pending_count(self) -> int:
        """Return accepted commands that have not completed."""
        return len(self._pending)

    @property
    def closed(self) -> bool:
        """Report whether process and arena ownership has been released."""
        return self._closed

    def try_submit(self, task: int, indices: Any, worker: int) -> bool:
        """Publish one Torch fetch unit and its indices through the native arena."""
        payload = pickle.dumps(indices, protocol=pickle.HIGHEST_PROTOCOL)
        accepted = self._resources.try_submit_command(
            task,
            COMPAT_STAGE,
            worker,
            payload,
        )
        if accepted:
            self._pending[task] = worker
        return accepted

    def try_receive(self) -> tuple[int, Any] | None:
        """Return one completion without imposing consumer order."""
        for worker in range(len(self._processes)):
            completion = self._resources.try_receive(worker)
            if completion is None:
                continue
            task, status, payload, _cost_ns = completion
            owner = self._pending.pop(task, None)
            if owner != worker:
                raise RuntimeError("compat completion lane does not match dispatch")
            if status != 0:
                reraise_worker_exception(payload, worker)
            return task, self._decoder.decode(payload, worker)
        return None

    def wait_for_completion(self, deadline: float | None) -> None:
        """Wait on lane signals while retaining liveness polls."""
        timeout = POLL_SECONDS
        if deadline is not None:
            timeout = max(0.0, min(timeout, deadline - time.monotonic()))
        for control in wait(self._controls, timeout):
            try:
                while control.poll():
                    if control.recv() != ("ready",):
                        raise RuntimeError("compat worker returned an invalid signal")
            except (BrokenPipeError, EOFError, OSError):
                pass
        self._check_liveness(deadline)

    def deadline(self) -> float | None:
        """Create the current delivery timeout deadline."""
        return None if self._timeout == 0 else time.monotonic() + self._timeout

    def drain(self) -> None:
        """Discard every admitted result before reusing persistent lanes."""
        deadline = self.deadline()
        while self._pending:
            if self.try_receive() is None:
                self.wait_for_completion(deadline)

    def close(self) -> None:
        """Stop every lane and release native arena ownership."""
        if self._closed:
            return
        self._closed = True
        for control in self._controls:
            try:
                control.send(("stop",))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self._processes:
            process.join(5.0)
            if process.is_alive():
                process.terminate()
                process.join(5.0)
        for control in self._controls:
            control.close()
        self._controls.clear()
        self._processes.clear()
        self._pending.clear()
        self._resources = None

    def _check_liveness(self, deadline: float | None) -> None:
        dead = [
            process.pid
            for process in self._processes
            if process.pid is not None and not process.is_alive()
        ]
        if dead:
            identifiers = ", ".join(str(pid) for pid in dead)
            self.close()
            raise RuntimeError(
                f"DataLoader worker (pid(s) {identifiers}) exited unexpectedly"
            )
        if deadline is not None and time.monotonic() >= deadline:
            self.close()
            raise RuntimeError(f"DataLoader timed out after {self._timeout} seconds")


def _resolve_context(requested: Any) -> Any:
    if requested is not None:
        return mp.get_context(requested) if isinstance(requested, str) else requested
    if sys.platform in {"win32", "darwin"}:
        return mp.get_context("spawn")
    return mp.get_context()

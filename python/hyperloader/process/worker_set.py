"""Operating-system process and control-pipe ownership for one pool."""

from __future__ import annotations

import multiprocessing as mp
import sys
from multiprocessing.connection import Connection
from typing import Any

from .serialization import encode_multiprocessing
from .worker import worker_main

SHUTDOWN_SECONDS = 5.0


def resolve_context(requested: Any) -> Any:
    """Resolve an explicit context or the spawn-safe platform default."""
    if requested is not None:
        return mp.get_context(requested) if isinstance(requested, str) else requested
    if sys.platform in {"win32", "darwin"}:
        return mp.get_context("spawn")
    return mp.get_context("forkserver")


class WorkerSet:
    """Own, attach, replace, and stop the pool's persistent processes."""

    def __init__(
        self,
        context: Any,
        dataset: Any,
        worker_init_fn: Any,
        worker_count: int,
        root_seed: int,
        completion_stride: int | None,
    ) -> None:
        self._context = context
        self._dataset = dataset
        self._worker_init_fn = worker_init_fn
        self._worker_count = worker_count
        self._root_seed = root_seed
        self._completion_stride = completion_stride
        self.controls: list[Connection] = []
        self.processes: list[mp.Process] = []

    def launch_all(self, probe: tuple[int, int, int]) -> None:
        """Launch every worker and give worker zero the construction probe."""
        for worker in range(self._worker_count):
            owner, process = self._launch(worker, probe if worker == 0 else None)
            self.controls.append(owner)
            self.processes.append(process)

    def attach_all(self, resources: Any, layout: Any) -> None:
        """Attach every live worker to its native transport and arena."""
        for worker, control in enumerate(self.controls):
            control.send(("attach", resources.descriptor(worker), layout))

    def replace(self, worker: int, resources: Any, layout: Any) -> None:
        """Replace one proven-dead worker after its native resources reset."""
        self.processes[worker].join(0)
        self.controls[worker].close()
        owner, process = self._launch(worker, None)
        self.controls[worker] = owner
        self.processes[worker] = process
        owner.send(("attach", resources.descriptor(worker), layout))

    def close(self) -> None:
        """Request graceful shutdown, then terminate only unresponsive workers."""
        for control in self.controls:
            try:
                control.send(("stop",))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(SHUTDOWN_SECONDS)
            if process.is_alive():
                process.terminate()
                process.join(SHUTDOWN_SECONDS)
        self.release()

    def abort(self) -> None:
        """Terminate every live worker immediately after a fatal pool event."""
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(SHUTDOWN_SECONDS)
        self.release()

    def release(self) -> None:
        """Close local control handles and discard process objects."""
        for control in self.controls:
            control.close()
        self.controls.clear()
        self.processes.clear()

    def _launch(
        self, worker: int, probe: tuple[int, int, int] | None
    ) -> tuple[Connection, mp.Process]:
        owner, child = self._context.Pipe(duplex=True)
        dataset_payload = encode_multiprocessing(
            (self._dataset, self._worker_init_fn)
        )
        process = self._context.Process(
            target=worker_main,
            args=(
                child,
                dataset_payload,
                worker,
                self._worker_count,
                self._root_seed,
                self._completion_stride,
                probe,
            ),
            daemon=True,
        )
        process.start()
        child.close()
        return owner, process

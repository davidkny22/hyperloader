"""Persistent process ownership and black-box request lifecycle."""

from __future__ import annotations

import time
from collections import deque
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any

from hyperloader import _hyperloader

from .batching import BatchLayout, decode_batch
from .exceptions import reraise_worker_exception
from .recovery import check_worker, restart_worker
from .serialization import ResultDecoder
from .worker import BLACK_BOX_STAGE
from .worker_set import WorkerSet, resolve_context

POLL_SECONDS = 0.005


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
        queue_capacity: int | None = 2,
        frontier_ceiling: int | None = None,
        frontier_minimum: int = 1,
        frontier_budget: int | None = None,
        on_worker_death: str = "close",
        batch_size: int | None = None,
        delivery_batch_size: int | None = None,
    ) -> None:
        context = resolve_context(multiprocessing_context)
        self._timeout = timeout
        self._closed = False
        self._next_worker = 0
        self._on_worker_death = on_worker_death
        self._batch_size = batch_size
        self._batch_layout: BatchLayout | None = None
        self._batch_shape: dict[str, object] = {"source": "probe-pending"}
        self._resources: Any = None
        self._worker_set = WorkerSet(
            context,
            dataset,
            worker_init_fn,
            worker_count,
            root_seed,
            batch_size,
            delivery_batch_size,
        )
        self._completion_signals = [0] * worker_count
        self._probe_key = (probe_epoch, probe_position, probe_index)
        self._probe_status = 0
        self._probe_payload: bytes | None = None
        self._probe_cost_ns = 1
        self._immediate: deque[tuple[int, int, int, bytes, int]] = deque()
        self._pending: dict[tuple[int, int], tuple[int, int, int]] = {}
        self._decoder = ResultDecoder()
        try:
            self._worker_set.launch_all(self._probe_key)
            status, probe_payload, layout, shape, probe_cost_ns = self._receive_probe()
            self._batch_layout = layout
            self._batch_shape = (
                {"source": "probe-pending"}
                if shape is None
                else {**shape, "source": "probe"}
            )
            if layout is None:
                self._batch_size = None
            self._probe_status = status
            self._bytes_sample = max(
                1, layout[2] if layout is not None else len(probe_payload)
            )
            if queue_capacity is None:
                if frontier_ceiling is None or frontier_budget is None:
                    raise ValueError(
                        "derived queue capacity requires a frontier ceiling and budget"
                    )
                affordable = frontier_budget // self._bytes_sample
                self._frontier_ceiling = max(
                    frontier_minimum, min(frontier_ceiling, affordable)
                )
                self._frontier_budget_bound = self._frontier_ceiling < frontier_ceiling
                from .sizing import queue_capacity as resolve_queue_capacity

                command_ceiling = (
                    (self._frontier_ceiling + self._batch_size - 1) // self._batch_size
                    if self._batch_size is not None
                    else self._frontier_ceiling
                )
                queue_capacity = resolve_queue_capacity(command_ceiling, worker_count)
            else:
                self._frontier_ceiling = frontier_ceiling or frontier_minimum
                self._frontier_budget_bound = False
            batch_capacity = (
                0
                if self._batch_size is None or layout is None
                else self._batch_size * layout[2] + 65_536
            )
            payload_capacity = max(262_144, len(probe_payload), batch_capacity)
            exception_capacity = max(65_536, len(probe_payload))
            self._resources = _hyperloader._ProcessResources(
                worker_count,
                queue_capacity,
                payload_capacity,
                exception_capacity,
                None if registry_path is None else Path(registry_path),
            )
            self._worker_set.attach_all(self._resources, self._batch_layout)
            self._probe_payload = probe_payload
            self._probe_cost_ns = probe_cost_ns
        except BaseException:
            self.close()
            raise

    @property
    def worker_pids(self) -> tuple[int, ...]:
        """Return stable process identifiers for liveness tests and diagnosis."""
        return tuple(
            process.pid
            for process in self._worker_set.processes
            if process.pid is not None
        )

    @property
    def worker_count(self) -> int:
        """Return the fixed number of persistent process workers."""
        return len(self._worker_set.processes)

    @property
    def batch_size(self) -> int | None:
        """Return the enabled homogeneous worker batch size."""
        return self._batch_size

    @property
    def bytes_sample(self) -> int:
        """Return the probe-measured sample payload size used by frontier budgets."""
        return self._bytes_sample

    @property
    def batch_shape_fingerprint(self) -> dict[str, object]:
        """Return the probe result without retaining the sampled value."""
        return self._batch_shape

    @property
    def frontier_ceiling(self) -> int:
        """Return the probe-frozen frontier ceiling in per-rank samples."""
        return self._frontier_ceiling

    @property
    def frontier_budget_bound(self) -> bool:
        """Return whether measured sample bytes reduced the formula ceiling."""
        return self._frontier_budget_bound

    @property
    def retained_probe_command(self) -> int | None:
        """Return the command position that must retain worker-zero routing."""
        if self._probe_payload is None:
            return None
        position = self._probe_key[1]
        return (
            position // self._batch_size if self._batch_size is not None else position
        )

    def try_submit(
        self,
        epoch: int,
        position: int,
        index: int,
        worker: int,
        *,
        batch_len: int = 0,
    ) -> bool:
        """Attempt one targeted black-box dispatch without waiting for capacity."""
        if self._closed:
            raise RuntimeError("process pool is closed")
        key = (epoch, position, index)
        if batch_len < 0:
            raise ValueError("batch length cannot be negative")
        if (
            self._probe_payload is not None
            and key == self._probe_key
            and batch_len == 0
        ):
            if worker != 0:
                raise RuntimeError("construction probe must retain worker zero routing")
            self._immediate.append(
                (
                    worker,
                    position,
                    self._probe_status,
                    self._probe_payload,
                    self._probe_cost_ns,
                )
            )
            self._probe_payload = None
            return True
        if self._probe_payload is not None and key == self._probe_key:
            self._probe_payload = None
        accepted = self._resources.try_submit(
            epoch, position, index, BLACK_BOX_STAGE, worker, batch_len
        )
        if accepted:
            self._pending[(worker, position)] = (epoch, index, batch_len)
        return accepted

    def try_receive(self, worker: int) -> tuple[int, int, bytes, int] | None:
        """Attempt one completion without imposing consumer delivery order."""
        if self._immediate and self._immediate[0][0] == worker:
            _, position, status, payload, cost_ns = self._immediate.popleft()
            return position, status, payload, cost_ns
        uses_completion_signals = self._batch_size is not None
        if uses_completion_signals and self._completion_signals[worker] == 0:
            return None
        completion = self._resources.try_receive(worker)
        if completion is None and uses_completion_signals:
            raise RuntimeError("worker signaled a completion before publishing it")
        if completion is None:
            return None
        if uses_completion_signals:
            self._completion_signals[worker] -= 1
        self._pending.pop((worker, completion[0]), None)
        return completion

    def decode(self, status: int, payload: bytes, worker: int) -> Any:
        """Reconstruct a successful sample or re-raise its worker exception."""
        if status == 0:
            return self._decoder.decode(payload, worker)
        reraise_worker_exception(payload, worker)

    def decode_batch(self, status: int, payload: Any, worker: int) -> Any:
        """Deliver one raw in-place batch or decode its compatibility fallback."""
        if status == 2:
            if self._batch_layout is None:
                raise RuntimeError("raw batch completion has no probed layout")
            return decode_batch(payload, self._batch_layout)
        return self.decode(status, payload, worker)

    def execute(self, epoch: int, position: int, index: int) -> Any:
        """Execute one black-box sample and reconstruct its result or exception."""
        if self._closed:
            raise RuntimeError("process pool is closed")
        worker = self._next_worker
        self._next_worker = (self._next_worker + 1) % self.worker_count
        deadline = None if self._timeout == 0 else time.monotonic() + self._timeout
        while not self.try_submit(epoch, position, index, worker):
            self._check_worker(worker, deadline)
            time.sleep(POLL_SECONDS)
        while True:
            completion = self.try_receive(worker)
            if completion is not None:
                received_position, status, payload, _cost_ns = completion
                if received_position != position:
                    raise RuntimeError(
                        "process completion position does not match dispatch"
                    )
                return self.decode(status, payload, worker)
            self._check_worker(worker, deadline)
            self.wait_for_completion(deadline)

    def deadline(self) -> float | None:
        """Create the current request's timeout deadline."""
        return None if self._timeout == 0 else time.monotonic() + self._timeout

    def check_workers(self, deadline: float | None) -> None:
        """Validate every process and the consumer timeout deadline."""
        for worker in range(self.worker_count):
            self._check_worker(worker, deadline)

    def wait_for_completion(self, deadline: float | None) -> bool:
        """Wait for a worker completion signal while retaining liveness polls."""
        timeout = POLL_SECONDS
        if deadline is not None:
            timeout = max(0.0, min(timeout, deadline - time.monotonic()))
        ready = wait(self._worker_set.controls, timeout)
        observed = False
        for worker, control in enumerate(self._worker_set.controls):
            if control not in ready:
                continue
            try:
                while control.poll():
                    message = control.recv()
                    if message != ("ready",):
                        raise RuntimeError(
                            "worker returned an invalid completion signal"
                        )
                    if self._batch_size is not None:
                        self._completion_signals[worker] += 1
                    observed = True
            except (BrokenPipeError, EOFError, OSError):
                self._check_worker(worker, deadline)
        return observed

    def close(self) -> None:
        """Stop workers, reclaim native resources, and make closure idempotent."""
        if self._closed:
            return
        self._closed = True
        self._worker_set.close()
        self._release_handles()

    def abort(self) -> None:
        """Terminate a timed-out pool immediately, then release native handles."""
        if self._closed:
            return
        self._closed = True
        self._worker_set.abort()
        self._release_handles()

    def _release_handles(self) -> None:
        """Close controls and drop native owners after every shutdown mode."""
        self._completion_signals.clear()
        self._pending.clear()
        self._resources = None

    def _receive_probe(
        self,
    ) -> tuple[int, bytes, BatchLayout | None, dict[str, object] | None, int]:
        while True:
            control = self._worker_set.controls[0]
            if control.poll(POLL_SECONDS):
                kind, status, payload, layout, shape, cost_ns = control.recv()
                if kind != "probe":
                    raise RuntimeError("worker returned an invalid probe response")
                return status, payload, layout, shape, cost_ns
            self._check_worker(0, None)

    def _restart_worker(self, worker: int) -> list[int]:
        return restart_worker(self, worker)

    def _check_worker(self, worker: int, deadline: float | None) -> None:
        check_worker(self, worker, deadline)

    def __del__(self) -> None:
        self.close()

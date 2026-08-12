"""Owner-side bounded queues and shared arena slots."""

from __future__ import annotations

import multiprocessing as mp
from collections import deque
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

from .command import WorkerCommand


class ProcessResources:
    """Own bounded queues and one shared arena for a process pool."""

    def __init__(
        self,
        worker_count: int,
        queue_capacity: int = 2,
        payload_capacity: int = 262_144,
        exception_capacity: int = 65_536,
        registry_path: Path | None = None,
    ) -> None:
        del registry_path
        if not 1 <= worker_count <= 1022:
            raise ValueError("worker_count must be between 1 and 1022")
        if queue_capacity <= 0:
            raise ValueError("queue capacity must be positive")
        if payload_capacity <= 0 or exception_capacity <= 0:
            raise ValueError("payload and exception capacities must be positive")
        self._worker_count = worker_count
        self._queue_capacity = queue_capacity
        self._payload_capacity = payload_capacity
        self._exception_capacity = exception_capacity
        self._slot_stride = payload_capacity + exception_capacity
        channels = [self._make_channels() for _ in range(worker_count)]
        self._dispatch_recv = [channel[0] for channel in channels]
        self._dispatch_send = [channel[1] for channel in channels]
        self._completion_recv = [channel[2] for channel in channels]
        self._completion_send = [channel[3] for channel in channels]
        self._arena = SharedMemory(
            create=True,
            size=worker_count * queue_capacity * self._slot_stride,
        )
        self._free = [deque(range(queue_capacity)) for _ in range(worker_count)]
        self._pending: dict[tuple[int, int], int] = {}
        self._closed = False

    def descriptor(self, worker: int) -> tuple[Any, ...]:
        """Return serializable attachment inputs for one worker."""
        self._validate_worker(worker)
        return (
            self._dispatch_recv[worker],
            self._completion_send[worker],
            self._arena.name,
            worker,
            self._queue_capacity,
            self._payload_capacity,
            self._exception_capacity,
        )

    def try_submit(
        self,
        epoch: int,
        position: int,
        index: int,
        stage_plan: int,
        worker: int,
        batch_len: int = 0,
    ) -> bool:
        """Reserve one slot and attempt a targeted dispatch."""
        return self._submit(
            WorkerCommand(position, epoch, index, stage_plan, worker, batch_len, -1),
            None,
        )

    def try_submit_command(
        self, position: int, stage_plan: int, worker: int, payload: bytes
    ) -> bool:
        """Write command metadata into its slot before dispatch."""
        if not payload:
            raise ValueError("command payload must not be empty")
        if len(payload) > self._payload_capacity:
            raise ValueError(
                f"command payload is {len(payload)} bytes but capacity is "
                f"{self._payload_capacity} bytes"
            )
        return self._submit(
            WorkerCommand(position, 0, len(payload), stage_plan, worker, 0, -1),
            payload,
        )

    def try_receive(self, worker: int) -> tuple[int, int, bytes, int] | None:
        """Copy one completed slot and release it to the bounded arena."""
        self._validate_worker(worker)
        completion = self._completion_recv[worker]
        if not completion.poll():
            return None
        position, status, produced, cost_ns, slot = completion.recv()
        expected = self._pending.pop((worker, position), None)
        if expected is None or expected != slot:
            raise RuntimeError("worker completed an unknown command coordinate")
        try:
            payload = self._read(worker, slot, exception=status == 1, length=produced)
        finally:
            self._free[worker].append(slot)
        return position, status, payload, cost_ns

    def reclaim_dead_worker(self, worker: int) -> list[int]:
        """Reclaim every slot only after its worker is proven dead."""
        self._validate_worker(worker)
        positions = sorted(
            position for owner, position in self._pending if owner == worker
        )
        for position in positions:
            slot = self._pending.pop((worker, position))
            if slot not in self._free[worker]:
                self._free[worker].append(slot)
        self._replace_channels(worker)
        return positions

    def restart_worker(self, worker: int) -> list[int]:
        """Reset a proven-dead worker's bounded transport."""
        return self.reclaim_dead_worker(worker)

    def close(self) -> None:
        """Release the shared arena and its manager process."""
        if self._closed:
            return
        self._closed = True
        try:
            self._arena.close()
            self._arena.unlink()
        finally:
            for channel in (
                self._dispatch_recv,
                self._dispatch_send,
                self._completion_recv,
                self._completion_send,
            ):
                for endpoint in channel:
                    endpoint.close()

    def _submit(self, command: WorkerCommand, payload: bytes | None) -> bool:
        if self._closed:
            raise RuntimeError("process resources are closed")
        worker = command.worker
        self._validate_worker(worker)
        key = (worker, command.position)
        if key in self._pending:
            raise ValueError("worker already has this position pending")
        if not self._free[worker]:
            return False
        slot = self._free[worker].popleft()
        command = WorkerCommand(
            command.position,
            command.epoch,
            command.index,
            command.stage_plan,
            command.worker,
            command.batch_len,
            slot,
        )
        if payload is not None:
            self._write(worker, slot, payload, exception=False)
        try:
            self._dispatch_send[worker].send(command)
        except (BrokenPipeError, EOFError, OSError):
            self._free[worker].appendleft(slot)
            return False
        self._pending[key] = slot
        return True

    def _offset(self, worker: int, slot: int, exception: bool) -> int:
        base = (worker * self._queue_capacity + slot) * self._slot_stride
        return base + (self._payload_capacity if exception else 0)

    def _write(
        self, worker: int, slot: int, payload: bytes, *, exception: bool
    ) -> None:
        capacity = self._exception_capacity if exception else self._payload_capacity
        if len(payload) > capacity:
            raise ValueError("payload exceeds its fallback arena slot")
        offset = self._offset(worker, slot, exception)
        self._arena.buf[offset : offset + len(payload)] = payload

    def _read(self, worker: int, slot: int, *, exception: bool, length: int) -> bytes:
        capacity = self._exception_capacity if exception else self._payload_capacity
        if not 0 <= length <= capacity:
            raise RuntimeError("completion length exceeds its fallback arena slot")
        offset = self._offset(worker, slot, exception)
        return bytes(self._arena.buf[offset : offset + length])

    def _validate_worker(self, worker: int) -> None:
        if not 0 <= worker < self._worker_count:
            raise ValueError(f"worker {worker} is out of range")

    @staticmethod
    def _make_channels() -> tuple[Any, Any, Any, Any]:
        dispatch_recv, dispatch_send = mp.Pipe(duplex=False)
        completion_recv, completion_send = mp.Pipe(duplex=False)
        return dispatch_recv, dispatch_send, completion_recv, completion_send

    def _replace_channels(self, worker: int) -> None:
        for channel in (
            self._dispatch_recv,
            self._dispatch_send,
            self._completion_recv,
            self._completion_send,
        ):
            channel[worker].close()
        dispatch_recv, dispatch_send, completion_recv, completion_send = (
            self._make_channels()
        )
        self._dispatch_recv[worker] = dispatch_recv
        self._dispatch_send[worker] = dispatch_send
        self._completion_recv[worker] = completion_recv
        self._completion_send[worker] = completion_send

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, BrokenPipeError, EOFError, FileNotFoundError, OSError):
            return

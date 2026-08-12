"""Worker-side attachment to fallback arena storage."""

from __future__ import annotations

from multiprocessing.shared_memory import SharedMemory
from typing import Any

from .command import WorkerCommand


class WorkerEndpoint:
    """Worker attachment to bounded dispatch, completion, and arena storage."""

    def __init__(
        self,
        dispatch: Any,
        completion: Any,
        arena_name: str,
        worker: int,
        queue_capacity: int,
        payload_capacity: int,
        exception_capacity: int,
    ) -> None:
        self._dispatch = dispatch
        self._completion = completion
        self._arena = SharedMemory(name=arena_name)
        self._worker = worker
        self._queue_capacity = queue_capacity
        self._payload_capacity = payload_capacity
        self._exception_capacity = exception_capacity
        self._slot_stride = payload_capacity + exception_capacity

    def try_recv(self) -> WorkerCommand | None:
        """Receive one dispatch without blocking control handling."""
        if not self._dispatch.poll():
            return None
        return self._dispatch.recv()

    def read_command(self, command: WorkerCommand) -> bytes:
        """Read owner-written command bytes."""
        return self._read(command, command.index, exception=False)

    def write_batch_row(
        self, command: WorkerCommand, row_offset: int, payload: memoryview
    ) -> None:
        """Write one contiguous row into its assigned batch slot."""
        data = bytes(payload)
        if row_offset < 0 or row_offset + len(data) > self._payload_capacity:
            raise ValueError("batch row exceeds its fallback arena slot")
        offset = self._offset(command, exception=False) + row_offset
        self._arena.buf[offset : offset + len(data)] = data

    def try_complete_ready(
        self, command: WorkerCommand, payload: bytes, cost_ns: int
    ) -> bool:
        """Write and publish one successful encoded payload."""
        self._write(command, payload, exception=False)
        return self._complete(command, 0, len(payload), cost_ns)

    def try_complete_batch(
        self, command: WorkerCommand, produced_length: int, cost_ns: int
    ) -> bool:
        """Publish one batch already written in place."""
        if not 0 <= produced_length <= self._payload_capacity:
            raise ValueError("batch length exceeds its fallback arena slot")
        return self._complete(command, 2, produced_length, cost_ns)

    def try_complete_exception(
        self, command: WorkerCommand, payload: bytes, cost_ns: int
    ) -> bool:
        """Write and publish one detached worker exception."""
        self._write(command, payload, exception=True)
        return self._complete(command, 1, len(payload), cost_ns)

    def close(self) -> None:
        """Release the worker mapping without unlinking owner storage."""
        self._dispatch.close()
        self._completion.close()
        self._arena.close()

    def _complete(
        self, command: WorkerCommand, status: int, produced: int, cost_ns: int
    ) -> bool:
        try:
            self._completion.send(
                (command.position, status, produced, cost_ns, command.slot)
            )
        except (BrokenPipeError, EOFError, OSError):
            return False
        return True

    def _offset(self, command: WorkerCommand, *, exception: bool) -> int:
        base = (self._worker * self._queue_capacity + command.slot) * self._slot_stride
        return base + (self._payload_capacity if exception else 0)

    def _write(
        self, command: WorkerCommand, payload: bytes, *, exception: bool
    ) -> None:
        capacity = self._exception_capacity if exception else self._payload_capacity
        if len(payload) > capacity:
            raise ValueError("payload exceeds its fallback arena slot")
        offset = self._offset(command, exception=exception)
        self._arena.buf[offset : offset + len(payload)] = payload

    def _read(self, command: WorkerCommand, length: int, *, exception: bool) -> bytes:
        capacity = self._exception_capacity if exception else self._payload_capacity
        if not 0 <= length <= capacity:
            raise ValueError("command length exceeds its fallback arena slot")
        offset = self._offset(command, exception=exception)
        return bytes(self._arena.buf[offset : offset + length])

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, BrokenPipeError, EOFError, FileNotFoundError, OSError):
            return

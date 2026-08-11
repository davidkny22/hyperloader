"""Pinned native ALU activity for Spark machine-state measurements."""

from __future__ import annotations

import ctypes
import os
import threading
import time
from pathlib import Path


class AluSpinner:
    """Run pure integer ALU loops on an explicit set of Linux CPUs."""

    def __init__(self, library: Path, cores: tuple[int, ...]) -> None:
        if not cores or len(set(cores)) != len(cores):
            raise ValueError("spinner cores must be a nonempty unique sequence")
        native = ctypes.CDLL(str(library))
        native.hyperloader_alu_spin.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        native.hyperloader_alu_spin.restype = ctypes.c_uint64
        native.hyperloader_alu_store_stop.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
        ]
        native.hyperloader_alu_store_stop.restype = None
        self._native = native
        self._cores = cores
        self._stop = ctypes.c_uint32(0)
        self._threads: list[threading.Thread] = []
        self._ready = [threading.Event() for _ in cores]
        self._records: list[dict[str, int | float]] = []
        self._started_at = 0.0
        self._stopped_at = 0.0

    def start(self) -> None:
        """Start one native loop per configured core and wait for every pin."""
        if self._threads:
            raise RuntimeError("spinner has already started")
        self._native.hyperloader_alu_store_stop(ctypes.byref(self._stop), 0)
        self._started_at = time.perf_counter()
        for index, core in enumerate(self._cores):
            thread = threading.Thread(
                target=self._run,
                args=(index, core),
                name=f"alu-spinner-{core}",
            )
            thread.start()
            self._threads.append(thread)
        for ready in self._ready:
            if not ready.wait(timeout=5.0):
                self.stop()
                raise RuntimeError(
                    "native ALU spinner did not start within five seconds"
                )

    def stop(self) -> dict[str, object]:
        """Stop every native loop and return its exact placement record."""
        if not self._threads:
            raise RuntimeError("spinner has not started")
        self._native.hyperloader_alu_store_stop(ctypes.byref(self._stop), 1)
        for thread in self._threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise RuntimeError(f"native ALU spinner {thread.name} did not stop")
        self._stopped_at = time.perf_counter()
        return {
            "process_id": os.getpid(),
            "cores": list(self._cores),
            "active_seconds": self._stopped_at - self._started_at,
            "threads": sorted(self._records, key=lambda item: int(item["core"])),
        }

    def _run(self, index: int, core: int) -> None:
        os.sched_setaffinity(0, {core})
        record: dict[str, int | float] = {
            "core": core,
            "native_thread_id": threading.get_native_id(),
            "started_seconds": time.perf_counter() - self._started_at,
        }
        self._ready[index].set()
        record["result"] = int(
            self._native.hyperloader_alu_spin(ctypes.byref(self._stop))
        )
        record["stopped_seconds"] = time.perf_counter() - self._started_at
        self._records.append(record)

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


class DutyCycleAluSpinner:
    """Run one native periodic ALU pulse inside the consumer process."""

    def __init__(
        self,
        library: Path,
        core: int,
        *,
        active_microseconds: int,
        period_microseconds: int,
    ) -> None:
        if active_microseconds <= 0 or active_microseconds > period_microseconds:
            raise ValueError("active time must be positive and no longer than the period")
        native = ctypes.CDLL(str(library))
        native.hyperloader_alu_pulse.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        native.hyperloader_alu_pulse.restype = ctypes.c_uint64
        native.hyperloader_alu_store_stop.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
        ]
        native.hyperloader_alu_store_stop.restype = None
        self._native = native
        self._core = core
        self._active_nanoseconds = active_microseconds * 1_000
        self._period_nanoseconds = period_microseconds * 1_000
        self._stop = ctypes.c_uint32(0)
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._record: dict[str, int | float] = {}
        self._started_at = 0.0

    def start(self) -> None:
        """Start the native pulse and wait until its thread is pinned."""
        if self._thread is not None:
            raise RuntimeError("duty-cycle spinner has already started")
        self._native.hyperloader_alu_store_stop(ctypes.byref(self._stop), 0)
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run,
            name=f"alu-pulse-{self._core}",
        )
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            self.stop()
            raise RuntimeError("native ALU pulse did not start within five seconds")

    def stop(self) -> dict[str, object]:
        """Stop the native pulse and return its exact duty and placement record."""
        if self._thread is None:
            raise RuntimeError("duty-cycle spinner has not started")
        self._native.hyperloader_alu_store_stop(ctypes.byref(self._stop), 1)
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("native ALU pulse did not stop")
        stopped_at = time.perf_counter()
        return {
            "process_id": os.getpid(),
            "core": self._core,
            "active_microseconds": self._active_nanoseconds / 1_000,
            "period_microseconds": self._period_nanoseconds / 1_000,
            "duty_percent": 100.0
            * self._active_nanoseconds
            / self._period_nanoseconds,
            "active_seconds": stopped_at - self._started_at,
            "thread": self._record,
        }

    def _run(self) -> None:
        os.sched_setaffinity(0, {self._core})
        self._record = {
            "native_thread_id": threading.get_native_id(),
            "started_seconds": time.perf_counter() - self._started_at,
        }
        self._ready.set()
        self._record["result"] = int(
            self._native.hyperloader_alu_pulse(
                ctypes.byref(self._stop),
                self._active_nanoseconds,
                self._period_nanoseconds,
            )
        )
        self._record["stopped_seconds"] = time.perf_counter() - self._started_at


class DutyCycleAluSpinnerGroup:
    """Run identical native pulses on an explicit set of Linux CPUs."""

    def __init__(
        self,
        library: Path,
        cores: tuple[int, ...],
        *,
        active_microseconds: int,
        period_microseconds: int,
    ) -> None:
        if not cores or len(set(cores)) != len(cores):
            raise ValueError("pulse cores must be a nonempty unique sequence")
        self._spinners = [
            DutyCycleAluSpinner(
                library,
                core,
                active_microseconds=active_microseconds,
                period_microseconds=period_microseconds,
            )
            for core in cores
        ]
        self._started = 0

    def start(self) -> None:
        """Start every pulse and stop the started subset if one fails."""
        try:
            for spinner in self._spinners:
                spinner.start()
                self._started += 1
        except BaseException:
            for spinner in reversed(self._spinners[: self._started]):
                spinner.stop()
            self._started = 0
            raise

    def stop(self) -> dict[str, object]:
        """Stop every pulse and return placement records in core order."""
        if self._started != len(self._spinners):
            raise RuntimeError("pulse group is not fully started")
        reports = [spinner.stop() for spinner in reversed(self._spinners)]
        self._started = 0
        reports.reverse()
        return {
            "cores": [int(report["core"]) for report in reports],
            "threads": reports,
        }

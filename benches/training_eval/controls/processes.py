"""Feeder-process evidence for paired training cells."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .host import affinity, thread_count
from .worker_probe import THREAD_ENVIRONMENT_KEYS


def process_record(pid: int) -> dict[str, Any]:
    """Capture one process's placement, thread count, and thread inputs."""
    return {
        "affinity": affinity(pid),
        "environment": _process_environment(pid),
        "os_thread_count": thread_count(f"/proc/{pid}/status"),
        "pid": pid,
    }


def worker_probe_records(
    directory: Path | None, *, expected_workers: int, timeout_seconds: float = 10.0
) -> list[dict[str, Any]]:
    """Read worker-boot records from one feeder-specific directory."""
    if directory is None or expected_workers == 0:
        return []
    deadline = time.monotonic() + timeout_seconds
    while True:
        records = _read_worker_probes(directory)
        if len(records) >= expected_workers or time.monotonic() >= deadline:
            return records
        time.sleep(0.01)


def validate_worker_probes(
    records: list[dict[str, Any]], *, expected_workers: int
) -> None:
    """Require one one-thread boot record per configured process worker."""
    if expected_workers == 0:
        if records:
            raise ValueError("zero-worker feeder produced worker environment records")
        return
    worker_ids = {record.get("worker_id") for record in records}
    expected_ids = set(range(expected_workers))
    if worker_ids != expected_ids or len(records) != expected_workers:
        raise ValueError("worker environment records do not match configured workers")
    if any(record.get("torch_intra_op_threads") != 1 for record in records):
        raise ValueError("process worker did not start with one Torch intra-op thread")


def _process_environment(pid: int) -> dict[str, str | None] | str:
    path = Path(f"/proc/{pid}/environ")
    if pid == os.getpid() or not path.exists():
        return {key: os.environ.get(key) for key in THREAD_ENVIRONMENT_KEYS}
    try:
        entries = path.read_bytes().split(b"\0")
    except OSError:
        return "unavailable"
    environment = {}
    decoded = {
        key.decode("utf-8", errors="replace"): value.decode(
            "utf-8", errors="replace"
        )
        for entry in entries
        if b"=" in entry
        for key, value in (entry.split(b"=", 1),)
    }
    for key in THREAD_ENVIRONMENT_KEYS:
        environment[key] = decoded.get(key)
    return environment


def _read_worker_probes(directory: Path) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    records = []
    for path in sorted(directory.glob("worker-*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("kind") != "training-worker-environment":
            raise ValueError(f"invalid worker environment record: {path}")
        records.append(document)
    return records

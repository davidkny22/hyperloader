"""Worker-local threading and affinity evidence."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
)


@dataclass(frozen=True)
class WorkerEnvironmentProbe:
    """Assert and persist the worker environment before timed execution."""

    output: str

    def __call__(self, worker_id: int) -> None:
        import torch

        intra_op = torch.get_num_threads()
        if intra_op != 1:
            raise RuntimeError(
                f"paired training workers require one Torch intra-op thread, observed {intra_op}"
            )
        output = Path(self.output)
        output.mkdir(parents=True, exist_ok=True)
        document = {
            "affinity": _affinity(),
            "environment": {
                key: os.environ.get(key) for key in THREAD_ENVIRONMENT_KEYS
            },
            "kind": "training-worker-environment",
            "os_thread_count": _thread_count(),
            "pid": os.getpid(),
            "torch_inter_op_threads": torch.get_num_interop_threads(),
            "torch_intra_op_threads": intra_op,
            "worker_id": worker_id,
        }
        descriptor, temporary = tempfile.mkstemp(
            dir=output, prefix=f"worker-{worker_id}-", suffix=".json.tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(document, handle, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            Path(temporary).replace(output / f"worker-{worker_id}-{os.getpid()}.json")
        finally:
            temporary_path = Path(temporary)
            if temporary_path.exists():
                temporary_path.unlink()


def _affinity() -> list[int] | str:
    if not hasattr(os, "sched_getaffinity"):
        return "unavailable"
    return sorted(os.sched_getaffinity(0))


def _thread_count() -> int | str:
    status = Path("/proc/self/status")
    if not status.exists():
        return "unavailable"
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("Threads:"):
            return int(line.split(":", 1)[1].strip())
    return "unavailable"

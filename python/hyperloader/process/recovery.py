"""Dead-worker recovery policy for persistent process pools."""

from __future__ import annotations

import time
from typing import Any

from .exceptions import WorkerDied
from .worker import BLACK_BOX_STAGE


def restart_worker(pool: Any, worker: int) -> list[int]:
    """Replace one dead process and replay every reclaimed reservation."""
    positions = sorted(pool._resources.restart_worker(worker))
    pool._worker_set.replace(worker, pool._resources, pool._batch_layout)
    pool._completion_signals[worker] = 0
    for position in positions:
        epoch, index, batch_len = pool._pending[(worker, position)]
        if not pool._resources.try_submit(
            epoch,
            position,
            index,
            BLACK_BOX_STAGE,
            worker,
            batch_len,
        ):
            raise RuntimeError("replacement worker transport rejected recovered work")
    return positions


def check_worker(pool: Any, worker: int, deadline: float | None) -> None:
    """Apply the configured death policy and enforce the consumer deadline."""
    process = pool._worker_set.processes[worker]
    if not process.is_alive():
        exitcode = process.exitcode
        positions: list[int] = []
        if pool._resources is not None:
            if pool._on_worker_death == "restart":
                positions = pool._restart_worker(worker)
                raise WorkerDied(
                    f"hyperloader worker {worker} exited with code {exitcode}; "
                    f"restarted after reclaiming positions {positions}"
                )
            positions = pool._resources.reclaim_dead_worker(worker)
        pool.abort()
        raise RuntimeError(
            f"hyperloader worker {worker} exited with code {exitcode}; "
            f"closed after reclaiming positions {positions}"
        )
    if deadline is not None and time.monotonic() >= deadline:
        pool.abort()
        raise RuntimeError(f"DataLoader timed out after {pool._timeout} seconds")

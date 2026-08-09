"""Construction-time process-pool preparation from a black-box plan."""

from __future__ import annotations

from typing import Any

from .pool import ProcessPool
from .sizing import frontier_depth, queue_capacity


def prepare_process_pool(loader: Any) -> None:
    """Spawn and probe the process tier once when its map plan is executable."""
    if loader._process_pool is not None or loader._plan is None or not loader._plan.length:
        return
    depth = frontier_depth(loader)
    probe_index = loader._plan.index(loader.root_seed, loader._epoch, 0)
    loader._process_pool = ProcessPool(
        loader.dataset,
        loader.num_workers,
        loader.root_seed,
        loader._epoch,
        0,
        probe_index,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context,
        timeout=loader.timeout,
        queue_capacity=queue_capacity(depth, loader.num_workers),
        on_worker_death=loader.config.executor.on_worker_death,
    )

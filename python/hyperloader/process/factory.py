"""Construction-time process-pool preparation from a black-box plan."""

from __future__ import annotations

from typing import Any

from .pool import ProcessPool
from .sizing import frontier_budget, frontier_ceiling, frontier_minimum


def prepare_process_pool(loader: Any) -> None:
    """Spawn and probe the process tier once when its map plan is executable."""
    if (
        loader._process_pool is not None
        or loader._plan is None
        or not loader._plan.length
    ):
        return
    depth = frontier_ceiling(loader)
    probe_index = loader._plan.index(loader.root_seed, loader._epoch, 0)
    batch_size = (
        loader.batch_size
        if (
            loader.config.executor.on_worker_death == "close"
            and not loader._plan.shuffle
            and loader.batch_size is not None
            and loader.batch_size > 1
        )
        else None
    )
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
        queue_capacity=None,
        frontier_ceiling=depth,
        frontier_minimum=frontier_minimum(loader),
        frontier_budget=frontier_budget(loader),
        on_worker_death=loader.config.executor.on_worker_death,
        batch_size=batch_size,
    )
    if loader._controller is None:
        from hyperloader.control import build_controller

        loader._controller = build_controller(loader)

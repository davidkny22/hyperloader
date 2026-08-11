"""Construction-time process-pool preparation from a black-box plan."""

from __future__ import annotations

from typing import Any

from ..stages import Pipeline
from .pool import ProcessPool
from .sizing import delivery_length, frontier_budget, frontier_ceiling, frontier_minimum


def prepare_process_pool(loader: Any) -> None:
    """Spawn and probe the process tier once when its map plan is executable."""
    if (
        loader._process_pool is not None
        or loader._plan is None
        or not loader._plan.length
        or (
            getattr(loader, "_sampler_runtime", None) is None
            and delivery_length(loader) == 0
        )
    ):
        return
    depth = frontier_ceiling(loader)
    sampler_runtime = getattr(loader, "_sampler_runtime", None)
    probe = None if sampler_runtime is None else sampler_runtime.probe()
    if sampler_runtime is not None and probe is None:
        return
    probe_position, probe_coordinate, probe_index = (
        (0, loader._map_coordinate(0), loader._map_index(loader._epoch, 0))
        if probe is None
        else (probe[0], probe[0], probe[1])
    )
    batch_size = (
        loader.batch_size
        if (
            loader.config.executor.on_worker_death == "close"
            and not loader._plan.shuffle
            and loader.batch_size is not None
            and loader.batch_size > 1
            and not isinstance(loader.dataset, Pipeline)
            and sampler_runtime is None
            and loader._map_placement.batch_transport_safe
        )
        else None
    )
    loader._process_pool = ProcessPool(
        loader._execution_dataset,
        loader.num_workers,
        loader.root_seed,
        loader._epoch,
        probe_position,
        probe_index,
        probe_coordinate=probe_coordinate,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context,
        timeout=loader.timeout,
        queue_capacity=None,
        frontier_ceiling=depth,
        frontier_minimum=frontier_minimum(loader),
        frontier_budget=frontier_budget(loader),
        on_worker_death=loader.config.executor.on_worker_death,
        batch_size=batch_size,
        delivery_batch_size=loader.batch_size,
    )
    if loader._controller is None:
        from hyperloader.control import build_controller

        loader._controller = build_controller(loader)

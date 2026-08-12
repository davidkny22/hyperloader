"""Native map-style execution for Torch-compatible worker lanes."""

from __future__ import annotations

import weakref
from typing import Any

from .lane_pool import CompatLanePool
from .lane_runtime import CompatLaneRuntime
from .multi_iterator import CompatMultiIterator
from .rng import capture_torch_source, draw_base_seed, restore_torch_source


def iterate_native_map(loader: Any) -> CompatMultiIterator:
    """Create a map iterator over hyperloader-owned lane transport."""
    active = (
        None if loader._active_iterator_ref is None else loader._active_iterator_ref()
    )
    pool = loader._compat_lane_pool
    retain_workers = loader.persistent_workers and pool is not None and not pool.closed
    if active is not None and not active.complete:
        active.invalidate(shutdown=not retain_workers)
    if retain_workers:
        pool.drain()
    elif pool is not None:
        pool.close()
        pool = None
        loader._compat_lane_pool = None

    resume = loader._resume_compat_multi_state
    if resume is None:
        iterator_generator = capture_torch_source(loader._compat_generator)
        reused_base_seed = bool(loader._compat_has_started and retain_workers)
        if pool is None:
            base_seed = draw_base_seed(loader._compat_generator)
            pool = CompatLanePool(
                loader,
                base_seed,
                {},
                capture_state=loader.config.determinism.compat_resume == "on",
            )
            loader._compat_lane_pool = pool
            loader._compat_base_seed = base_seed
        else:
            base_seed = loader._compat_base_seed
            if base_seed is None:
                raise RuntimeError("persistent compat lanes lost their base seed")
        runtime = CompatLaneRuntime(loader, pool)
        wrapper = CompatMultiIterator(
            loader,
            runtime,
            iterator_generator,
            tagged=loader.config.determinism.compat_resume == "on",
            reused_base_seed=reused_base_seed,
            base_seed=base_seed,
        )
    else:
        restore_torch_source(loader._compat_generator, resume.iterator_generator)
        if resume.reused_base_seed:
            base_seed = resume.base_seed
            if base_seed is None:
                raise RuntimeError("compat checkpoint lost its reused base seed")
        else:
            base_seed = draw_base_seed(loader._compat_generator)
            if resume.base_seed is not None and base_seed != resume.base_seed:
                raise ValueError("compat checkpoint base seed does not reproduce")
        _validate_lane_seeds(base_seed, resume.lane_seeds)
        pool = CompatLanePool(
            loader,
            base_seed,
            resume.lane_states,
            capture_state=True,
        )
        loader._compat_lane_pool = pool
        loader._compat_base_seed = base_seed
        runtime = CompatLaneRuntime(
            loader,
            pool,
            skip=resume.delivered_batches,
            phase=resume.assignment_phase,
            prefetch=resume.sampler_position - resume.delivered_batches,
        )
        wrapper = CompatMultiIterator(
            loader,
            runtime,
            resume.iterator_generator,
            delivered=resume.delivered_batches,
            reused_base_seed=resume.reused_base_seed,
            base_seed=base_seed,
        )
        restore_torch_source(loader._compat_generator, resume.current_generator)
    loader._resume_compat_multi_state = None
    loader._compat_has_started = True
    loader._active_iterator_ref = weakref.ref(wrapper)
    return wrapper


def _validate_lane_seeds(base_seed: int, lane_seeds: dict[int, int]) -> None:
    for lane, seed in lane_seeds.items():
        if seed != base_seed + lane:
            raise ValueError("compat checkpoint lane seed does not match base seed")

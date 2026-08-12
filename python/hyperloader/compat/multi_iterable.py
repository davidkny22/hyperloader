"""Native iterable execution for Torch-compatible worker lanes."""

from __future__ import annotations

import weakref
from typing import Any

from .lane_pool import CompatLanePool
from .lane_runtime import CompatLaneRuntime
from .multi_iterator import CompatMultiIterator
from .rng import capture_torch_source, draw_base_seed


def iterate_native_iterable(loader: Any) -> CompatMultiIterator:
    """Create an iterable iterator over hyperloader-owned lane transport."""
    active = (
        None if loader._active_iterator_ref is None else loader._active_iterator_ref()
    )
    pool = loader._compat_lane_pool
    retain_workers = loader.persistent_workers and pool is not None and not pool.closed
    if active is not None and not active.complete:
        active.invalidate(shutdown=not retain_workers)
    iterator_generator = capture_torch_source(loader._compat_generator)
    if retain_workers:
        pool.drain()
        pool.reset_iterable()
        base_seed = loader._compat_base_seed
        if base_seed is None:
            raise RuntimeError("persistent compat lanes lost their base seed")
    else:
        if pool is not None:
            pool.close()
        base_seed = draw_base_seed(loader._compat_generator)
        pool = CompatLanePool(
            loader,
            base_seed,
            {},
            capture_state=False,
        )
        loader._compat_lane_pool = pool
        loader._compat_base_seed = base_seed
    wrapper = CompatMultiIterator(
        loader,
        CompatLaneRuntime(loader, pool),
        iterator_generator,
        tagged=False,
        reused_base_seed=retain_workers,
        base_seed=base_seed,
    )
    loader._resume_compat_multi_state = None
    loader._compat_has_started = True
    loader._active_iterator_ref = weakref.ref(wrapper)
    return wrapper

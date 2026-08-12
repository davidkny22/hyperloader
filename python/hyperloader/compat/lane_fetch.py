"""Dataset fetcher construction and execution inside compat lanes."""

from __future__ import annotations

from typing import Any

from torch.utils.data._utils.fetch import (
    _IterableDatasetFetcher,
    _MapDatasetFetcher,
)

from hyperloader.process.worker import encode_exception

from .protocol import LaneExhausted, TaggedBatch
from .worker import capture_worker_state


def build_fetcher(
    dataset: Any,
    auto_collation: bool,
    collate_fn: Any,
    drop_last: bool,
    iterable: bool,
) -> Any:
    """Construct Torch's exact map or iterable fetcher for one lane."""
    fetcher_type = _IterableDatasetFetcher if iterable else _MapDatasetFetcher
    return fetcher_type(dataset, auto_collation, collate_fn, drop_last)


def fetch_batch(
    fetcher: Any,
    indices: Any,
    batch: int,
    worker: int,
    seed: int,
    capture_state: bool,
    iterable: bool,
) -> tuple[int, Any]:
    """Fetch one unit and preserve iterable exhaustion as a control value."""
    try:
        state = capture_worker_state() if capture_state else b""
        value = fetcher.fetch(indices)
        if capture_state:
            value = TaggedBatch(batch, worker, seed, value, state)
        return 0, value
    except StopIteration as error:
        if iterable:
            return 0, LaneExhausted(worker)
        return encode_exception(error)
    except BaseException as error:
        return encode_exception(error)

"""Process-owned user collation transport and evaluation."""

from __future__ import annotations

import pickle
import time
from collections.abc import Callable
from typing import Any

from hyperloader.rng import _user_code_context

USER_COLLATE_STAGE = 1
COMMAND_RETRY_SECONDS = 0.005


def next_process_batch(iterator: Any) -> Any:
    """Execute the next native-order batch through process-owned collation."""
    start = iterator._position
    width = iterator._loader.batch_size or 1
    stop = min(iterator._length, start + width)
    entries = tuple(
        (
            iterator._loader._map_coordinate(position),
            iterator._loader._map_index(iterator._epoch, position),
        )
        for position in range(start, stop)
    )
    ordinal = iterator._delivered.base
    value = iterator._loader._process_pool.execute_collated(
        iterator._epoch,
        ordinal,
        entries,
        auto_collation=iterator._loader.batch_size is not None,
    )
    iterator._position = stop
    iterator._delivered.mark(ordinal)
    iterator._loader._epoch_state.mark_delivered(iterator._epoch)
    return value


def collated_command(
    loader: Any, epoch: int, batch_ordinal: int, length: int
) -> tuple[tuple[tuple[int, Any], ...], bool]:
    """Build one exact user-collation command for frontier dispatch."""
    width = loader.batch_size or 1
    start = batch_ordinal * width
    stop = min(length, start + width)
    entries = tuple(
        (
            loader._map_coordinate(position),
            loader._map_index(epoch, position),
        )
        for position in range(start, stop)
    )
    return entries, loader.batch_size is not None


def encode_collated_command(
    epoch: int,
    batch_ordinal: int,
    entries: tuple[tuple[int, Any], ...],
    *,
    auto_collation: bool,
) -> bytes:
    """Encode one user-collation command for native transport."""
    return pickle.dumps((epoch, batch_ordinal, entries, auto_collation), protocol=5)


def execute_collated(
    pool: Any,
    epoch: int,
    batch_ordinal: int,
    entries: tuple[tuple[int, Any], ...],
    *,
    auto_collation: bool,
) -> Any:
    """Submit one exact user-collation batch to a process worker."""
    if pool._closed:
        raise RuntimeError("process pool is closed")
    worker = pool._next_worker
    pool._next_worker = (pool._next_worker + 1) % pool.worker_count
    payload = encode_collated_command(
        epoch,
        batch_ordinal,
        entries,
        auto_collation=auto_collation,
    )
    deadline = pool.deadline()
    while not pool._resources.try_submit_command(
        batch_ordinal, USER_COLLATE_STAGE, worker, payload
    ):
        pool._check_worker(worker, deadline)
        time.sleep(COMMAND_RETRY_SECONDS)
    pool._pending[(worker, batch_ordinal)] = (epoch, 0, len(entries), batch_ordinal)
    while True:
        completion = pool.try_receive(worker)
        if completion is not None:
            position, status, result, _cost_ns = completion
            if position != batch_ordinal:
                raise RuntimeError(
                    "process completion position does not match dispatch"
                )
            return pool.decode(status, result, worker)
        pool._check_worker(worker, deadline)
        pool.wait_for_completion(deadline)


def evaluate_user_collate(
    dispatch: Any,
    dataset: Any,
    worker_id: int,
    worker_count: int,
    root_seed: int,
    encoder: Any,
    rng_context: Any,
    endpoint: Any,
    collate_fn: Any,
    evaluate_sample: Callable[..., tuple[int, Any]],
    encode_exception: Callable[[BaseException], tuple[int, bytes]],
) -> tuple[int, bytes, int | None, int]:
    """Evaluate one sample group and its user collation in one worker."""
    if collate_fn is None:
        status, payload = encode_exception(
            RuntimeError("user collation command has no callable")
        )
        return status, payload, None, 0
    try:
        epoch, ordinal, entries, auto_collation = pickle.loads(
            endpoint.read_command(dispatch)
        )
        values = []
        for coordinate, index in entries:
            status, value = evaluate_sample(
                dataset,
                worker_id,
                worker_count,
                root_seed,
                epoch,
                coordinate,
                index,
                rng_context,
            )
            if status != 0:
                return status, value, None, 0
            values.append(value)
        rng_context.install_collate(root_seed, epoch, ordinal)
        with _user_code_context(rng_context.current_sample):
            result = collate_fn(values if auto_collation else values[0])
        return 0, encoder.encode_uncached(result), None, 0
    except BaseException as error:  # noqa: BLE001
        status, payload = encode_exception(error)
        return status, payload, None, 0

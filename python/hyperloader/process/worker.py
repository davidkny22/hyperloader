"""Spawn-safe black-box worker loop."""

from __future__ import annotations

import pickle
import time
import traceback
from multiprocessing.connection import Connection
from typing import Any

from hyperloader import _hyperloader

from .batching import encode_batch, supports_worker_batch
from .parent_watchdog import start_parent_watchdog
from .rng import WorkerRngContext
from .serialization import ResultEncoder

BLACK_BOX_STAGE = 0
NO_PROBE_VALUE = object()


def worker_main(
    control: Connection,
    dataset_payload: bytes,
    worker_id: int,
    worker_count: int,
    root_seed: int,
    completion_stride: int | None,
    probe: tuple[int, int, int] | None,
) -> None:
    """Run one persistent dataset copy until the owner sends stop."""
    start_parent_watchdog()
    dataset, worker_init_fn = pickle.loads(dataset_payload)
    encoder = ResultEncoder()
    probe_value: Any = NO_PROBE_VALUE
    rng_context = WorkerRngContext(worker_id, worker_count, dataset)
    try:
        startup_error = None
        try:
            if worker_init_fn is not None:
                worker_init_fn(worker_id)
        except BaseException as error:
            startup_error = encode_exception(error)
        if probe is not None:
            if startup_error is None:
                epoch, position, index = probe
                status, result = evaluate_sample(
                    dataset,
                    worker_id,
                    worker_count,
                    root_seed,
                    epoch,
                    position,
                    index,
                    rng_context,
                )
                if status == 0:
                    probe_value = result
                    payload = encoder.encode(result)
                else:
                    payload = result
            else:
                status, payload = startup_error
            batch_supported = status == 0 and supports_worker_batch(probe_value)
            control.send(("probe", status, payload, batch_supported))
        command = control.recv()
        if command[0] == "stop":
            return
        if command[0] != "attach":
            raise RuntimeError("worker received an invalid control command")
        endpoint = _hyperloader._WorkerEndpoint(*command[1])
        probe_values = [] if probe_value is NO_PROBE_VALUE else [probe_value]
        run_commands(
            control,
            endpoint,
            dataset,
            worker_id,
            worker_count,
            root_seed,
            startup_error,
            encoder,
            probe_values,
            rng_context,
            completion_stride,
        )
    finally:
        rng_context.clear()
        control.close()


def run_commands(
    control: Connection,
    endpoint: Any,
    dataset: Any,
    worker_id: int,
    worker_count: int,
    root_seed: int,
    startup_error: tuple[int, bytes] | None,
    encoder: ResultEncoder,
    probe_values: list[Any],
    rng_context: WorkerRngContext,
    completion_stride: int | None,
) -> None:
    """Poll control and dispatch channels without blocking shutdown."""
    while True:
        if control.poll():
            command = control.recv()
            if command[0] == "stop":
                return
            raise RuntimeError("worker received an invalid control command")
        dispatch = endpoint.try_recv()
        if dispatch is None:
            time.sleep(0.0005)
            continue
        if startup_error is not None:
            status, payload = startup_error
            result = (status, payload)
        elif dispatch.stage_plan != BLACK_BOX_STAGE:
            status, payload = encode_exception(
                RuntimeError(f"unknown stage plan {dispatch.stage_plan}")
            )
            result = (status, payload)
        else:
            result = evaluate_dispatch(
                dispatch,
                dataset,
                worker_id,
                worker_count,
                root_seed,
                encoder,
                probe_values,
                rng_context,
            )
        publish_completion(
            control,
            endpoint,
            dispatch,
            result[0],
            result[1],
            completion_stride,
        )


def evaluate_dispatch(
    dispatch: Any,
    dataset: Any,
    worker_id: int,
    worker_count: int,
    root_seed: int,
    encoder: ResultEncoder,
    probe_values: list[Any],
    rng_context: WorkerRngContext,
) -> tuple[int, bytes]:
    """Evaluate one sample command or one contiguous default-collation batch."""
    if dispatch.batch_len == 0:
        status, value = evaluate_sample(
            dataset,
            worker_id,
            worker_count,
            root_seed,
            dispatch.epoch,
            dispatch.position,
            dispatch.index,
            rng_context,
        )
        return (0, encoder.encode(value)) if status == 0 else (status, value)

    values = []
    for offset in range(dispatch.batch_len):
        position = dispatch.index + offset
        if position == 0 and probe_values:
            values.append(probe_values.pop())
            continue
        status, value = evaluate_sample(
            dataset,
            worker_id,
            worker_count,
            root_seed,
            dispatch.epoch,
            position,
            position,
            rng_context,
        )
        if status != 0:
            return status, value
        values.append(value)
    try:
        return 0, encode_batch(values, encoder)
    except BaseException as error:
        return encode_exception(error)


def evaluate_sample(
    dataset: Any,
    worker_id: int,
    worker_count: int,
    root_seed: int,
    epoch: int,
    position: int,
    index: int,
    rng_context: WorkerRngContext,
) -> tuple[int, Any]:
    """Run one black-box sample under its exact RNG and worker view."""
    try:
        rng_context.install(root_seed, epoch, position)
        return 0, dataset[index]
    except BaseException as error:
        return encode_exception(error)


def encode_exception(error: BaseException) -> tuple[int, bytes]:
    """Detach exception identity and formatted traceback from live frames."""
    payload = (
        type(error).__module__,
        type(error).__qualname__,
        str(error),
        traceback.format_exc(),
    )
    return 1, pickle.dumps(payload, protocol=5)


def publish_completion(
    control: Connection,
    endpoint: Any,
    dispatch: Any,
    status: int,
    payload: bytes,
    completion_stride: int | None,
) -> None:
    """Retry bounded completion publication while preserving shutdown."""
    while True:
        complete = (
            endpoint.try_complete_ready(dispatch, payload)
            if status == 0
            else endpoint.try_complete_exception(dispatch, payload)
        )
        if complete:
            batch_boundary = (
                completion_stride is not None
                and (dispatch.position + 1) % completion_stride == 0
            )
            if dispatch.batch_len or batch_boundary:
                control.send(("ready",))
            return
        if control.poll():
            command = control.recv()
            if command[0] == "stop":
                return
            raise RuntimeError("worker received an invalid control command")
        time.sleep(0.0005)

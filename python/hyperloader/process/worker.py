"""Spawn-safe black-box worker loop."""

from __future__ import annotations

import pickle
import time
import traceback
from multiprocessing.connection import Connection
from typing import Any

from hyperloader import _hyperloader

from .batching import ArrayBatcher
from .parent_watchdog import start_parent_watchdog
from .rng import clear_worker_info, install_sample_rng, set_worker_info
from .serialization import ResultEncoder

BLACK_BOX_STAGE = 0
NO_PROBE_VALUE = object()


def worker_main(
    control: Connection,
    dataset_payload: bytes,
    worker_id: int,
    worker_count: int,
    root_seed: int,
    probe: tuple[int, int, int] | None,
    batch_size: int | None,
) -> None:
    """Run one persistent dataset copy until the owner sends stop."""
    start_parent_watchdog()
    dataset, worker_init_fn = pickle.loads(dataset_payload)
    encoder = ResultEncoder()
    probe_value: Any = NO_PROBE_VALUE
    set_worker_info(worker_id, worker_count, None, dataset)
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
                )
                if status == 0:
                    probe_value = result
                    payload = encoder.encode(result)
                else:
                    payload = result
            else:
                status, payload = startup_error
            control.send(("probe", status, payload))
        command = control.recv()
        if command[0] == "stop":
            return
        if command[0] != "attach":
            raise RuntimeError("worker received an invalid control command")
        endpoint = _hyperloader._WorkerEndpoint(*command[1])
        batcher = ArrayBatcher(batch_size, encoder)
        if probe_value is not NO_PROBE_VALUE:
            batcher.seed_probe(probe_value)
        run_commands(
            control,
            endpoint,
            dataset,
            worker_id,
            worker_count,
            root_seed,
            startup_error,
            batcher,
        )
    finally:
        clear_worker_info()
        control.close()


def run_commands(
    control: Connection,
    endpoint: Any,
    dataset: Any,
    worker_id: int,
    worker_count: int,
    root_seed: int,
    startup_error: tuple[int, bytes] | None,
    batcher: ArrayBatcher,
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
            completions = batcher.failure(dispatch, status, payload)
        elif dispatch.stage_plan != BLACK_BOX_STAGE:
            status, payload = encode_exception(
                RuntimeError(f"unknown stage plan {dispatch.stage_plan}")
            )
            completions = batcher.failure(dispatch, status, payload)
        else:
            status, result = evaluate_sample(
                dataset,
                worker_id,
                worker_count,
                root_seed,
                dispatch.epoch,
                dispatch.position,
                dispatch.index,
            )
            completions = (
                batcher.success(dispatch, result)
                if status == 0
                else batcher.failure(dispatch, status, result)
            )
        for completion in completions:
            publish_completion(
                control,
                endpoint,
                completion.dispatch,
                completion.status,
                completion.payload,
            )


def evaluate_sample(
    dataset: Any,
    worker_id: int,
    worker_count: int,
    root_seed: int,
    epoch: int,
    position: int,
    index: int,
) -> tuple[int, Any]:
    """Run one black-box sample under its exact RNG and worker view."""
    try:
        torch_seed = install_sample_rng(root_seed, epoch, position)
        set_worker_info(worker_id, worker_count, torch_seed, dataset)
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
) -> None:
    """Retry bounded completion publication while preserving shutdown."""
    while True:
        complete = (
            endpoint.try_complete_ready(dispatch, payload)
            if status == 0
            else endpoint.try_complete_exception(dispatch, payload)
        )
        if complete:
            return
        if control.poll():
            command = control.recv()
            if command[0] == "stop":
                return
            raise RuntimeError("worker received an invalid control command")
        time.sleep(0.0005)

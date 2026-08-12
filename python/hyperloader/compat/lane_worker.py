"""Torch-compatible worker execution over native arena transport."""

from __future__ import annotations

import inspect
import pickle
import random
import time
from multiprocessing.connection import Connection
from typing import Any

import numpy as np
import torch
from torch.utils.data._utils.worker import _generate_state

from hyperloader import _hyperloader
from hyperloader.process.parent_watchdog import start_parent_watchdog
from hyperloader.process.serialization import ResultEncoder
from hyperloader.process.worker import encode_exception

from .lane_fetch import build_fetcher, fetch_batch
from .worker import restore_worker_state

COMPAT_STAGE = 1


def lane_worker_main(
    control: Connection,
    payload: bytes | tuple[Any, Any, Any, bool, bool, bool],
    worker: int,
    workers: int,
    base_seed: int,
    capture_state: bool,
    restored_state: bytes | None,
) -> None:
    """Run one Torch-seeded dataset copy behind a native worker endpoint."""
    start_parent_watchdog()
    (
        dataset,
        collate_fn,
        worker_init_fn,
        auto_collation,
        drop_last,
        iterable,
    ) = pickle.loads(payload) if isinstance(payload, bytes) else payload
    seed = base_seed + worker
    startup_error = _initialize_lane(
        dataset,
        worker_init_fn,
        worker,
        workers,
        base_seed,
        seed,
        restored_state,
    )
    fetcher_spec = (dataset, auto_collation, collate_fn, drop_last, iterable)
    fetcher = build_fetcher(*fetcher_spec)
    encoder = ResultEncoder()
    try:
        command = control.recv()
        if command[0] == "stop":
            return
        if command[0] != "attach":
            raise RuntimeError("compat worker received an invalid attach command")
        endpoint = _hyperloader._WorkerEndpoint(*command[1])
        _run_lane(
            control,
            endpoint,
            fetcher,
            fetcher_spec,
            encoder,
            worker,
            seed,
            capture_state,
            iterable,
            startup_error,
        )
    finally:
        _clear_worker_info()
        control.close()


def _initialize_lane(
    dataset: Any,
    worker_init_fn: Any,
    worker: int,
    workers: int,
    base_seed: int,
    seed: int,
    restored_state: bytes | None,
) -> tuple[int, bytes] | None:
    try:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(_generate_state(base_seed, worker))
        _install_worker_info(dataset, worker, workers, seed)
        if worker_init_fn is not None:
            worker_init_fn(worker)
        if restored_state is not None:
            restore_worker_state(restored_state)
    except BaseException as error:
        return encode_exception(error)
    return None


def _run_lane(
    control: Connection,
    endpoint: Any,
    fetcher: Any,
    fetcher_spec: tuple[Any, bool, Any, bool, bool],
    encoder: ResultEncoder,
    worker: int,
    seed: int,
    capture_state: bool,
    iterable: bool,
    startup_error: tuple[int, bytes] | None,
) -> None:
    while True:
        if control.poll():
            message = control.recv()
            if message[0] == "stop":
                return
            if message[0] == "reset":
                fetcher = build_fetcher(*fetcher_spec)
                control.send(("reset_ready",))
                continue
            raise RuntimeError("compat worker received an invalid control command")
        dispatch = endpoint.try_recv()
        if dispatch is None:
            time.sleep(0.0005)
            continue
        started = time.perf_counter_ns()
        if dispatch.stage_plan != COMPAT_STAGE:
            status, result = encode_exception(
                RuntimeError(f"unknown compat stage plan {dispatch.stage_plan}")
            )
        elif startup_error is not None:
            status, result = startup_error
        else:
            indices = pickle.loads(endpoint.read_command(dispatch))
            status, result = fetch_batch(
                fetcher,
                indices,
                dispatch.position,
                worker,
                seed,
                capture_state,
                iterable,
            )
        cost_ns = max(1, time.perf_counter_ns() - started)
        _publish(endpoint, dispatch, encoder, status, result, cost_ns)
        control.send(("ready",))


def _publish(
    endpoint: Any,
    dispatch: Any,
    encoder: ResultEncoder,
    status: int,
    result: Any,
    cost_ns: int,
) -> None:
    if status != 0:
        while not endpoint.try_complete_exception(dispatch, result, cost_ns):
            time.sleep(0.0005)
        return
    payload = encoder.encode(result)
    while not endpoint.try_complete_ready(dispatch, payload, cost_ns):
        time.sleep(0.0005)


def _install_worker_info(dataset: Any, worker: int, workers: int, seed: int) -> None:
    import torch.utils.data._utils.worker as worker_module

    fields = {
        "id": worker,
        "num_workers": workers,
        "seed": seed,
        "dataset": dataset,
        "rng": None,
        "worker_method": "multiprocessing",
    }
    signature = inspect.signature(worker_module.WorkerInfo)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        fields = {
            name: fields[name] for name in ("id", "num_workers", "seed", "dataset")
        }
    else:
        fields = {
            name: value
            for name, value in fields.items()
            if name in signature.parameters
        }
    worker_module._worker_info = worker_module.WorkerInfo(**fields)


def _clear_worker_info() -> None:
    import torch.utils.data._utils.worker as worker_module

    worker_module._worker_info = None

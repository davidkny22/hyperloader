"""One uninterrupted torch-versus-hyperloader identity cell."""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Any

from benchmark_protocol import (
    CommonConfig,
    EnvironmentMetadata,
    PairedObservation,
    SystemRun,
    TuningBudget,
)
from identity_feeders import (
    EQUAL_FRONTIER_DEPTH,
    HyperloaderFeeder,
    TorchFeeder,
    WORKERS,
)
from identity_workloads import make_identity_dataset
from overhead_environment import ClockSampler
from overhead_feeders import BATCH_SIZE, SAMPLE_WIDTH, resident_batch_count
from overhead_workload import GpuWorkload

WORKLOAD_REGIMES = {
    "fixed-text": "compute",
    "numpy-array": "bandwidth",
    "arrow-tabular": "bandwidth",
}


def run_identity_cell(
    *,
    ordinal: int,
    workload: str,
    environment: EnvironmentMetadata,
    llc_bytes: int,
    half_seconds: float = 45.0,
) -> dict[str, Any]:
    """Run one paired identity comparison and return raw protocol evidence."""
    if half_seconds <= 0:
        raise ValueError("half duration must be positive")
    try:
        regime = WORKLOAD_REGIMES[workload]
    except KeyError as error:
        raise ValueError(f"unknown identity workload {workload!r}") from error
    batch_bytes = BATCH_SIZE * SAMPLE_WIDTH * 8
    resident_batches = resident_batch_count(llc_bytes, batch_bytes)
    dataset = make_identity_dataset(workload, resident_batches)
    torch_feeder: TorchFeeder | None = None
    hyperloader_feeder: HyperloaderFeeder | None = None
    sampler = ClockSampler()
    original_affinity = os.sched_getaffinity(0)
    try:
        if ordinal % 2 == 0:
            torch_feeder = TorchFeeder(dataset)
            hyperloader_feeder = HyperloaderFeeder(dataset)
        else:
            hyperloader_feeder = HyperloaderFeeder(dataset)
            torch_feeder = TorchFeeder(dataset)
        import torch

        torch_batch = torch_feeder.next_batch()
        hyperloader_batch = hyperloader_feeder.next_batch()
        if not torch.equal(torch_batch, hyperloader_batch):
            raise RuntimeError("identity feeders produced different batch values")
        torch_feeder.warm()
        hyperloader_feeder.warm()
        os.sched_setaffinity(0, {19})
        gpu_workload = GpuWorkload(regime)
        gpu_workload.warm(torch_batch)
        order = (
            ("counterfactual", "loader")
            if ordinal % 2 == 0
            else ("loader", "counterfactual")
        )
        feeders = {
            "counterfactual": torch_feeder,
            "loader": hyperloader_feeder,
        }
        start_batches = {name: feeder.batches for name, feeder in feeders.items()}
        counts = {name: 0 for name in order}
        spills = 0
        sampler.start()
        started = time.perf_counter()
        midpoint = started + half_seconds
        finished = midpoint + half_seconds
        while (selected_at := time.perf_counter()) < finished:
            selected = order[0] if selected_at < midpoint else order[1]
            batch = feeders[selected].next_batch()
            gpu_workload.run(batch)
            completed_at = time.perf_counter()
            observed = order[0] if completed_at < midpoint else order[1]
            counts[observed] += 1
            if observed != selected:
                spills += 1
        clock_samples = sampler.stop()
    finally:
        if sampler._thread.is_alive():
            sampler.stop()
        os.sched_setaffinity(0, original_affinity)
        if hyperloader_feeder is not None:
            hyperloader_feeder.close()
        if torch_feeder is not None:
            torch_feeder.close()

    if torch_feeder is None or hyperloader_feeder is None:
        raise RuntimeError("identity feeder construction did not complete")

    config = CommonConfig(
        workload=workload,
        gpu_regime=regime,
        batch_size=BATCH_SIZE,
        workers=WORKERS,
        prefetch_depth=EQUAL_FRONTIER_DEPTH,
        delivery="host-sync-h2d",
        batch_shape=f"int64[{BATCH_SIZE},{SAMPLE_WIDTH}]",
        cache_regime="warm",
    )
    tuning = TuningBudget(0, 0.0, ())
    runs = {
        name: SystemRun(
            system=name,
            throughput=counts[name] / half_seconds,
            duration_seconds=half_seconds,
            warmed=True,
            config=config,
            tuning=tuning,
            environment=environment,
        )
        for name in order
    }
    observation = PairedObservation(
        ordinal=ordinal,
        first=runs[order[0]],
        second=runs[order[1]],
        uninterrupted=True,
    )
    return {
        **asdict(observation),
        "raw": {
            "systems": {"counterfactual": "torch", "loader": "hyperloader"},
            "clock_samples": clock_samples,
            "feeder_batches": {
                name: feeder.batches - start_batches[name]
                for name, feeder in feeders.items()
            },
            "startup_seconds": {
                "torch": torch_feeder.startup_seconds,
                "hyperloader": hyperloader_feeder.startup_seconds,
            },
            "boundary_spill_operations": spills,
            "llc_bytes": llc_bytes,
            "resident_batches": resident_batches,
            "resident_bytes": resident_batches * batch_bytes,
            "stage_plan_pin": "black-box process; native default collation; strict order",
        },
    }

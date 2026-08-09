"""One uninterrupted paired feeder-swap cell on Spark."""

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
    process_transport_split,
)
from overhead_environment import ClockSampler
from overhead_feeders import (
    BATCH_SIZE,
    FRONTIER_DEPTH,
    WORKERS,
    FixedTextDataset,
    LoaderFeeder,
    ResidentFeeder,
    payload_sizes,
    resident_batch_count,
)
from overhead_workload import GpuWorkload


def run_cell(
    *,
    ordinal: int,
    regime: str,
    environment: EnvironmentMetadata,
    llc_bytes: int,
    half_seconds: float = 45.0,
) -> dict[str, Any]:
    """Run one alternating feeder pair and return protocol plus raw evidence."""
    if half_seconds <= 0:
        raise ValueError("half duration must be positive")
    batch_bytes = BATCH_SIZE * 512 * 8
    resident_batches = resident_batch_count(llc_bytes, batch_bytes)
    dataset = FixedTextDataset(resident_batches)
    resident = ResidentFeeder(dataset)
    loader = LoaderFeeder(dataset)
    logical_bytes, serialized_bytes, batch_bytes = payload_sizes(dataset)
    original_affinity = os.sched_getaffinity(0)
    sampler = ClockSampler()
    try:
        loader.warm()
        resident.warm()
        os.sched_setaffinity(0, {19})
        workload = GpuWorkload(regime)
        workload.warm(resident.next_batch())
        order = ("counterfactual", "loader") if ordinal % 2 == 0 else (
            "loader",
            "counterfactual",
        )
        feeders = {"counterfactual": resident, "loader": loader}
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
            workload.run(batch)
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
        loader.close()

    config = CommonConfig(
        workload="fixed-text",
        gpu_regime=regime,
        batch_size=BATCH_SIZE,
        workers=WORKERS,
        prefetch_depth=FRONTIER_DEPTH,
        delivery="host-sync-h2d",
        batch_shape=f"int64[{BATCH_SIZE},512]",
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
    loader_batches = loader.batches - start_batches["loader"]
    split = process_transport_split(
        duration_seconds=half_seconds,
        samples=loader_batches * BATCH_SIZE,
        batches=loader_batches,
        logical_sample_bytes=logical_bytes,
        serialized_sample_bytes=serialized_bytes,
        batch_bytes=batch_bytes,
    )
    return {
        **asdict(observation),
        "raw": {
            "clock_samples": clock_samples,
            "feeder_batches": {
                name: feeder.batches - start_batches[name]
                for name, feeder in feeders.items()
            },
            "boundary_spill_operations": spills,
            "llc_bytes": llc_bytes,
            "resident_batches": resident_batches,
            "resident_bytes": resident_batches * batch_bytes,
            "byte_split": asdict(split),
        },
    }

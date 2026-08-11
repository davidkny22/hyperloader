"""One uninterrupted hyperloader-versus-reference GPU cell."""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Any

from benchmark_protocol import EnvironmentMetadata, TuningBudget
from dominance_feeders import build_feeder
from dominance_protocol import (
    DominanceObservation,
    DominanceRun,
    SelectedConfig,
)
from dominance_tuning import WARM_BATCHES
from dominance_workloads import WorkloadBundle
from overhead_environment import ClockSampler
from overhead_workload import GpuWorkload


def run_dominance_cell(
    *,
    ordinal: int,
    reference: str,
    workload: WorkloadBundle,
    selected: dict[str, SelectedConfig],
    tuning: TuningBudget,
    environment: EnvironmentMetadata,
    half_seconds: float = 45.0,
) -> dict[str, Any]:
    """Run one alternating feeder pair against a continuous GPU workload."""
    if reference not in {"torch", "spdl"}:
        raise ValueError("reference must be torch or spdl")
    if half_seconds <= 0:
        raise ValueError("half duration must be positive")
    order = (
        ("hyperloader", reference) if ordinal % 2 == 0 else (reference, "hyperloader")
    )
    feeders: dict[str, Any] = {}
    sampler = ClockSampler()
    original_affinity = os.sched_getaffinity(0)
    try:
        for system in order:
            feeders[system] = build_feeder(system, workload, selected[system])
        first_batches = {
            system: feeder.next_batch() for system, feeder in feeders.items()
        }
        import torch

        if not torch.equal(first_batches["hyperloader"], first_batches[reference]):
            raise RuntimeError("comparison feeders produced different batch values")
        for feeder in feeders.values():
            for _ in range(WARM_BATCHES):
                feeder.next_batch()
        os.sched_setaffinity(0, {max(original_affinity)})
        gpu_workload = GpuWorkload(workload.gpu_regime)
        gpu_workload.warm(first_batches["hyperloader"])
        start_batches = {system: feeder.batches for system, feeder in feeders.items()}
        counts = {system: 0 for system in order}
        spills = 0
        sampler.start()
        started = time.perf_counter()
        midpoint = started + half_seconds
        finished = midpoint + half_seconds
        while (selected_at := time.perf_counter()) < finished:
            system = order[0] if selected_at < midpoint else order[1]
            batch = feeders[system].next_batch()
            gpu_workload.run(batch)
            completed_at = time.perf_counter()
            observed = order[0] if completed_at < midpoint else order[1]
            counts[observed] += 1
            if observed != system:
                spills += 1
        clock_samples = sampler.stop()
        feeder_reports = {system: feeder.report() for system, feeder in feeders.items()}
    finally:
        if sampler._thread.is_alive():
            sampler.stop()
        os.sched_setaffinity(0, original_affinity)
        for feeder in feeders.values():
            feeder.close()

    runs = {
        system: DominanceRun(
            system=system,
            reference=reference,
            workload=workload.name,
            gpu_regime=workload.gpu_regime,
            throughput=counts[system] / half_seconds,
            duration_seconds=half_seconds,
            warmed=True,
            selected=selected[system],
            tuning=tuning,
            environment=environment,
        )
        for system in order
    }
    observation = DominanceObservation(
        ordinal=ordinal,
        first=runs[order[0]],
        second=runs[order[1]],
        uninterrupted=True,
    )
    return {
        **asdict(observation),
        "raw": {
            "boundary_spill_operations": spills,
            "clock_samples": clock_samples,
            "feeder_batches": {
                system: feeder.batches - start_batches[system]
                for system, feeder in feeders.items()
            },
            "feeder_reports": feeder_reports,
            "stage_plan_pin": workload.stage_plan_pin,
            "systems": {"loader": "hyperloader", "reference": reference},
        },
    }

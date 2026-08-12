"""Counted equal-budget tuning for dominance references."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from benchmark_protocol import TuningBudget
from dominance_feeders import build_feeder
from dominance_protocol import SelectedConfig
from dominance_workloads import WorkloadBundle

TUNING_CANDIDATES = (
    SelectedConfig(2, 2),
    SelectedConfig(2, 4),
    SelectedConfig(4, 2),
    SelectedConfig(4, 4),
    SelectedConfig(8, 2),
    SelectedConfig(8, 4),
)
TUNING_SECONDS = 2.0
WARM_BATCHES = 8


def tuning_budget(*, smoke: bool = False) -> TuningBudget:
    """Return the identical counted allowance assigned to every system."""
    trials = 1 if smoke else len(TUNING_CANDIDATES)
    seconds = 0.25 if smoke else TUNING_SECONDS
    return TuningBudget(
        trials=trials,
        wall_seconds=trials * seconds,
        knobs=("workers", "prefetch_factor"),
    )


def tune(
    system: str,
    workload: WorkloadBundle,
    *,
    worker_cpus: tuple[int, ...],
    smoke: bool = False,
) -> tuple[SelectedConfig, dict[str, Any]]:
    """Select the highest standalone batch rate under the fixed search grid."""
    candidates = TUNING_CANDIDATES[:1] if smoke else TUNING_CANDIDATES
    seconds = 0.25 if smoke else TUNING_SECONDS
    trials = []
    for selected in candidates:
        feeder = build_feeder(
            system,
            workload,
            selected,
            worker_cpus=worker_cpus,
        )
        try:
            for _ in range(WARM_BATCHES):
                feeder.next_batch()
            count = 0
            started = time.perf_counter()
            deadline = started + seconds
            while time.perf_counter() < deadline:
                feeder.next_batch()
                count += 1
            elapsed = time.perf_counter() - started
        finally:
            feeder.close()
        trials.append(
            {
                "selected": asdict(selected),
                "batches": count,
                "elapsed_seconds": elapsed,
                "batches_per_second": count / elapsed,
            }
        )
    winner = max(trials, key=lambda trial: trial["batches_per_second"])
    return SelectedConfig(**winner["selected"]), {
        "budget": asdict(tuning_budget(smoke=smoke)),
        "system": system,
        "trials": trials,
        "winner": winner,
    }

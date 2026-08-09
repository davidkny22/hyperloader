"""Acceptance checks and aggregation for Spark overhead evidence."""

from __future__ import annotations

import statistics
from typing import Any


def clock_samples_valid(samples: list[dict[str, Any]]) -> bool:
    """Require positive current-clock observations while the GPU is loaded."""
    loaded = [
        int(sample["clock_mhz"])
        for sample in samples
        if int(sample["utilization_percent"]) > 0
    ]
    return bool(loaded) and all(clock > 0 for clock in loaded)


def summarize_splits(cells: list[dict[str, Any]]) -> dict[str, float | str]:
    """Average equal-duration byte rates and state the instrumentation bound."""
    splits = [cell["raw"]["byte_split"] for cell in cells]
    if not splits:
        raise ValueError("split summary requires at least one cell")
    keys = (
        "model_input_gbps",
        "irreducible_host_gbps",
        "explicit_overhead_gbps",
        "explicit_total_host_gbps",
    )
    result: dict[str, float | str] = {
        key: statistics.fmean(float(split[key]) for split in splits) for key in keys
    }
    result["overhead_scope"] = (
        "exact explicit native copies; Python serialization copies are excluded"
    )
    result["stage_plan_pin"] = (
        "fixed-text black-box process; native default collation; host synchronous H2D"
    )
    return result

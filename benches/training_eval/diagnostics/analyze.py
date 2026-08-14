"""GIL-holding stack analysis for training diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from benches.dominance_pyspy_analyze import parse_raw

HALF_MARKERS = {
    "counterfactual": "profile_counterfactual_half",
    "hyperloader": "profile_hyperloader_half",
}
STAGE_MARKERS = {
    "next_batch": "profile_next_batch",
    "copy": "profile_copy",
    "compute": "profile_compute",
    "sync": "profile_sync",
}


def analyze_gil_profile(
    path: Path, *, total_seconds_by_system: dict[str, float]
) -> dict[str, Any]:
    """Summarize GIL-holding samples by feeder, stage, thread, and leaf."""
    rows = parse_raw(path)
    halves = {}
    for system, marker in HALF_MARKERS.items():
        selected = [
            (frames, weight)
            for frames, weight in rows
            if any(marker in frame for frame in frames)
        ]
        total = sum(weight for _frames, weight in selected)
        stages: Counter[str] = Counter()
        threads: Counter[str] = Counter()
        leaves: Counter[str] = Counter()
        for frames, weight in selected:
            stage = next(
                (
                    name
                    for name, stage_marker in STAGE_MARKERS.items()
                    if any(stage_marker in frame for frame in frames)
                ),
                "other",
            )
            stages[stage] += weight
            threads[frames[0]] += weight
            leaves[frames[-1]] += weight
        seconds = total_seconds_by_system[system]
        halves[system] = {
            "gil_samples": total,
            "gil_samples_per_second": total / seconds,
            "stage_samples": _counter_rows(stages, total),
            "thread_samples": _counter_rows(threads, total, limit=20),
            "leaf_samples": _counter_rows(leaves, total, limit=30),
        }
    return {"kind": "training-gil-profile", "source": str(path), "systems": halves}


def _counter_rows(
    values: Counter[str], total: int, *, limit: int | None = None
) -> list[dict[str, float | int | str]]:
    return [
        {
            "name": name,
            "samples": count,
            "percent": 0.0 if total == 0 else 100.0 * count / total,
        }
        for name, count in values.most_common(limit)
    ]


def main() -> None:
    """Join one raw GIL profile to its diagnostic half durations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    diagnostic = json.loads(arguments.diagnostic.read_text(encoding="utf-8"))
    totals = {
        system: sum(float(half["elapsed_seconds"]) for half in diagnostic["halves"])
        for system in HALF_MARKERS
    }
    report = analyze_gil_profile(arguments.input, total_seconds_by_system=totals)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

"""Extract half-specific clock residency from definitive dominance cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dominance_gpu_segments import split_clock_samples, summarize_clocks


def extract_residency(path: Path) -> dict[str, object]:
    """Aggregate ordered clock samples from an existing JSONL campaign."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("clock extraction requires at least one cell")
    aggregate: dict[str, list[dict[str, float | int]]] = {}
    cells = []
    for row in rows:
        half_seconds = float(row["first"]["duration_seconds"])
        order = (row["first"]["system"], row["second"]["system"])
        samples = row["raw"]["clock_samples"]
        cells.append(
            {
                "ordinal": row["ordinal"],
                "order": list(order),
                "halves": split_clock_samples(samples, order, half_seconds),
            }
        )
        for sample in samples:
            elapsed = float(sample["elapsed_seconds"])
            index = 0 if elapsed < half_seconds else 1
            relative = elapsed - index * half_seconds
            aggregate.setdefault(order[index], []).append(
                {**sample, "relative_seconds": relative}
            )
    return {
        "source": str(path),
        "cells": cells,
        "aggregate": {
            name: summarize_clocks(samples, half_seconds)
            for name, samples in aggregate.items()
        },
    }


def main() -> None:
    """Write one durable clock-residency extraction report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = extract_residency(arguments.input)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()

"""Summarize threaded py-spy raw profiles for paired consumer halves."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

HALF_MARKERS = {
    "hyperloader": "profile_hyperloader_half",
    "torch": "profile_torch_half",
}
STAGE_MARKERS = {
    "next_batch": "profile_next_batch",
    "copy": "profile_copy",
    "launch": "profile_launch",
    "sync": "profile_sync",
}
AUXILIARY_THREAD_MARKERS = ("alu-spinner-", "clock-sampler")


def parse_raw(path: Path) -> list[tuple[list[str], int]]:
    """Parse py-spy's collapsed raw stacks and sample weights."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        stack, separator, weight = line.rpartition(" ")
        if not separator:
            raise ValueError(f"raw profile line has no weight: {line!r}")
        rows.append((stack.split(";"), int(weight)))
    return rows


def analyze(path: Path, system: str | None = None) -> dict[str, object]:
    """Report half, stage, thread, leaf, and inclusive-frame sample shares."""
    result: dict[str, object] = {"source": str(path), "halves": {}}
    rows = parse_raw(path)
    selected_halves = HALF_MARKERS if system is None else {system: None}
    for half, marker in selected_halves.items():
        selected = (
            rows
            if marker is None
            else [
                (frames, weight)
                for frames, weight in rows
                if any(marker in frame for frame in frames)
            ]
        )
        ignored = sum(
            weight
            for frames, weight in selected
            if any(marker in frames[0] for marker in AUXILIARY_THREAD_MARKERS)
        )
        if system is not None:
            selected = [
                (frames, weight)
                for frames, weight in selected
                if not any(marker in frames[0] for marker in AUXILIARY_THREAD_MARKERS)
            ]
        total = sum(weight for _frames, weight in selected)
        stages = Counter()
        threads = Counter()
        leaves = Counter()
        inclusive = Counter()
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
            for frame in set(frames):
                inclusive[frame] += weight
        result["halves"][half] = {
            "samples": total,
            "excluded_auxiliary_samples": ignored if system is not None else 0,
            "stage_samples": _counter_rows(stages, total),
            "thread_samples": _counter_rows(threads, total, limit=20),
            "leaf_samples": _counter_rows(leaves, total, limit=30),
            "inclusive_samples": _counter_rows(inclusive, total, limit=40),
        }
    return result


def _counter_rows(
    values: Counter[str], total: int, *, limit: int | None = None
) -> list[dict[str, float | int | str]]:
    rows = [
        {
            "name": name,
            "samples": count,
            "percent": 0.0 if total == 0 else 100.0 * count / total,
        }
        for name, count in values.most_common(limit)
    ]
    return rows


def main() -> None:
    """Write one JSON summary for a threaded raw py-spy profile."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system", choices=tuple(HALF_MARKERS))
    arguments = parser.parse_args()
    report = analyze(arguments.input, arguments.system)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["halves"], sort_keys=True))


if __name__ == "__main__":
    main()

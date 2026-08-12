"""Measure passive observer cost through installed public loader paths."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import hyperloader
import torch
from hyperloader import DataLoader, diagnose

from observer_overhead_report import MINIMUM_PAIRS

BATCH_SIZE = 64


def _loader(kind: str, dataset: torch.Tensor) -> Any:
    if kind == "hyperloader":
        return DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2, seed=71)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        persistent_workers=True,
    )


def _close(loader: Any, iterator: Any) -> None:
    close = getattr(loader, "close", None)
    if callable(close):
        close()
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def _half(kind: str, dataset: torch.Tensor, observed: bool) -> dict[str, int]:
    loader = _loader(kind, dataset)
    iterator = iter(loader)
    observe = (lambda: diagnose(loader)) if observed else (lambda: None)
    checksum = 0
    batches = 0
    next(iterator)
    started_cpu = time.process_time_ns()
    started_wall = time.perf_counter_ns()
    try:
        for batch in iterator:
            checksum += int(batch[0].item()) + int(batch[-1].item())
            batches += 1
            if batches == dataset.shape[0] // BATCH_SIZE // 2:
                report = observe()
                if (
                    report is not None
                    and report.record["observation_mode"] != "passive"
                ):
                    raise AssertionError("observer did not use the passive path")
        wall_ns = time.perf_counter_ns() - started_wall
        cpu_ns = time.process_time_ns() - started_cpu
    finally:
        _close(loader, iterator)
    return {
        "batches": batches,
        "checksum": checksum,
        "cpu_ns": cpu_ns,
        "wall_ns": wall_ns,
    }


def _pair(
    kind: str,
    dataset: torch.Tensor,
    left_observed: bool,
    right_observed: bool,
    order: str,
) -> dict[str, Any]:
    left = _half(kind, dataset, left_observed)
    right = _half(kind, dataset, right_observed)
    if left["batches"] != right["batches"]:
        raise AssertionError("paired executions delivered different batch counts")
    return {
        "left_checksum": left["checksum"],
        "left_cpu_ns": left["cpu_ns"],
        "left_wall_ns": left["wall_ns"],
        "order": order,
        "right_checksum": right["checksum"],
        "right_cpu_ns": right["cpu_ns"],
        "right_wall_ns": right["wall_ns"],
    }


def _probe() -> dict[str, Any]:
    loader = torch.utils.data.DataLoader(torch.arange(256), batch_size=BATCH_SIZE)
    report = diagnose(loader, probe=True, probe_batches=4)
    probe = report.record["probe"]
    return {
        "consumed_batches": probe["consumed_batches"],
        "elapsed_ns": probe["elapsed_ns"],
        "gil_release_fraction": probe["gil_release_fraction"],
        "requested_batches": 4,
    }


def run_measurement(pairs: int, batches: int, expected_root: Path) -> dict[str, Any]:
    if pairs < MINIMUM_PAIRS:
        raise ValueError("pair count is below the measurement floor")
    package_root = Path(hyperloader.__file__).resolve().parent.parent
    if package_root != expected_root.resolve():
        raise RuntimeError("benchmark did not import the expected installed artifact")
    dataset = torch.arange((batches + 1) * BATCH_SIZE, dtype=torch.int64)
    loaders: dict[str, Any] = {}
    for kind in ("hyperloader", "torch"):
        _half(kind, dataset, False)
        _half(kind, dataset, True)
        measured = []
        noise = []
        for index in range(pairs):
            observed_first = index % 2 == 0
            measured.append(
                _pair(
                    kind,
                    dataset,
                    observed_first,
                    not observed_first,
                    "observer-first" if observed_first else "baseline-first",
                )
            )
            noise.append(_pair(kind, dataset, False, False, "null"))
        loaders[kind] = {"noise_pairs": noise, "pairs": measured}
    return {
        "active_probe": _probe(),
        "loaders": loaders,
        "metadata": {
            "batch_size": BATCH_SIZE,
            "batches_per_half": batches,
            "pair_count": pairs,
            "platform": platform.platform(),
            "public_path_verified": True,
            "python": sys.version,
            "torch": torch.__version__,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=MINIMUM_PAIRS)
    parser.add_argument("--batches", type=int, default=512)
    parser.add_argument("--expected-install-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run_measurement(
        arguments.pairs, arguments.batches, arguments.expected_install_root
    )
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

"""Run installed native-free fallback comparisons against stock Torch."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from fallback_no_regression_report import evaluate_report
from fallback_workloads import WORKLOADS, WorkloadSpec


def _stage_fallback(destination: Path) -> Path:
    spec = importlib.util.find_spec("hyperloader")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("an installed hyperloader package is required")
    source = Path(next(iter(spec.submodule_search_locations))).resolve()
    package = destination / "hyperloader"
    shutil.copytree(source, package)
    removed = []
    for pattern in ("*.so", "*.pyd", "*.dylib"):
        for artifact in package.rglob(pattern):
            artifact.unlink()
            removed.append(artifact.name)
    if not removed:
        raise RuntimeError("the installed package contained no native extension to sever")
    return package


def _run_child(arguments: argparse.Namespace, root: Path) -> dict[str, Any]:
    output = root / "raw.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--batch-size",
        str(arguments.batch_size),
        "--measured-batches",
        str(arguments.measured_batches),
        "--pairs",
        str(arguments.pairs),
        "--prefetch-factor",
        str(arguments.prefetch_factor),
        "--warmup-batches",
        str(arguments.warmup_batches),
        "--workers",
        str(arguments.workers),
        "--output",
        str(output),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(root), environment.get("PYTHONPATH", ""))
        if value
    )
    completed = subprocess.run(
        command,
        check=False,
        cwd=Path(__file__).resolve().parent,
        env=environment,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"fallback benchmark child exited {completed.returncode}")
    return json.loads(output.read_text(encoding="utf-8"))


def _loader(system: str, workload: WorkloadSpec, arguments: argparse.Namespace) -> Any:
    length = arguments.batch_size * (
        arguments.warmup_batches + arguments.measured_batches + 1
    )
    dataset = workload.dataset(length)
    common = {
        "batch_size": arguments.batch_size,
        "drop_last": True,
        "num_workers": arguments.workers,
        "persistent_workers": True,
        "prefetch_factor": arguments.prefetch_factor,
        "shuffle": False,
    }
    if system == "torch":
        import torch

        return torch.utils.data.DataLoader(dataset, **common)
    from hyperloader import DataLoader, HyperConfig
    from hyperloader.config import SchedulerConfig

    frontier = (
        arguments.batch_size * arguments.workers * arguments.prefetch_factor
    )
    return DataLoader(
        dataset,
        config=HyperConfig(
            scheduler=SchedulerConfig(frontier_depth=frontier, profile_cache="off")
        ),
        seed=1701,
        **common,
    )


def _measure(
    system: str, workload: WorkloadSpec, arguments: argparse.Namespace
) -> dict[str, int]:
    loader = _loader(system, workload, arguments)
    iterator = iter(loader)
    try:
        for _ in range(arguments.warmup_batches):
            next(iterator)
        start = time.perf_counter_ns()
        last = None
        for _ in range(arguments.measured_batches):
            last = next(iterator)
        elapsed = time.perf_counter_ns() - start
        if last is None:
            raise RuntimeError("the measurement delivered no batch")
        checksum = workload.checksum(last)
    finally:
        close = getattr(loader, "close", None)
        if close is not None:
            close()
    return {
        "checksum": checksum,
        "elapsed_ns": elapsed,
        "samples": arguments.batch_size * arguments.measured_batches,
    }


def _child_report(arguments: argparse.Namespace) -> dict[str, Any]:
    import hyperloader
    import torch
    from hyperloader import DataLoader, _hyperloader

    fallback_resolved = bool(getattr(_hyperloader, "IS_FALLBACK", False))
    public_path_verified = DataLoader.__module__.startswith("hyperloader")
    workloads: dict[str, list[dict[str, Any]]] = {}
    for workload in WORKLOADS:
        pairs = []
        for ordinal in range(arguments.pairs):
            order = ("fallback", "torch") if ordinal % 2 == 0 else ("torch", "fallback")
            runs = {system: _measure(system, workload, arguments) for system in order}
            pairs.append(
                {
                    "fallback": runs["fallback"],
                    "order": f"{order[0]}-first",
                    "ordinal": ordinal,
                    "torch": runs["torch"],
                }
            )
        workloads[workload.name] = pairs
    return {
        "metadata": {
            "batch_size": arguments.batch_size,
            "equal_tuning": True,
            "fallback_resolved": fallback_resolved,
            "hyperloader": hyperloader.__version__,
            "measured_batches": arguments.measured_batches,
            "pair_count": arguments.pairs,
            "prefetch_factor": arguments.prefetch_factor,
            "public_path_verified": public_path_verified,
            "python": sys.version,
            "torch": torch.__version__,
            "warmup_batches": arguments.warmup_batches,
            "workers": arguments.workers,
        },
        "workloads": workloads,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--measured-batches", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--warmup-batches", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.child:
        raw = _child_report(arguments)
        arguments.output.write_text(
            json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return
    with tempfile.TemporaryDirectory(prefix="hyperloader-fallback-bench-") as directory:
        root = Path(directory)
        _stage_fallback(root)
        raw = _run_child(arguments, root)
    decision = evaluate_report(
        raw,
        mutate=os.environ.get("HYPERLOADER_FALLBACK_BENCH_MUTATE") == "slow-fallback",
    )
    document = {"decision": decision, "raw": raw}
    arguments.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, sort_keys=True))
    if decision["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

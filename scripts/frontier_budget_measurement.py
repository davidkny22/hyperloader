"""Measure adaptive-frontier timing outside the permanent test suite."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

from hyperloader import DataLoader, HyperConfig, _hyperloader
from hyperloader.config import SchedulerConfig
from hyperloader.process import iterator as iterator_module


class PathologicalCostDataset:
    """Expose a repeatable heavy tail across otherwise inexpensive samples."""

    def __len__(self) -> int:
        return 64

    def __getitem__(self, index: int) -> int:
        time.sleep(0.008 if index in {11, 26, 41, 56} else 0.00025)
        return index


def _unbounded_depth(loader: DataLoader) -> int:
    """Plant a missing ceiling clamp in the public iterator path."""
    return loader._process_pool.frontier_ceiling * 2


def main() -> None:
    """Run the host-timed frontier measurement and write its raw metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--expected-install-root", type=Path, required=True)
    arguments = parser.parse_args()
    if (
        not Path(_hyperloader.__file__)
        .resolve()
        .is_relative_to(arguments.expected_install_root.resolve())
    ):
        raise RuntimeError("measurement did not import the requested installation")
    with tempfile.TemporaryDirectory() as directory:
        loader = DataLoader(
            PathologicalCostDataset(),
            batch_size=1,
            num_workers=4,
            seed=101,
            config=HyperConfig(
                scheduler=SchedulerConfig(profile_cache=Path(directory))
            ),
        )
        mutation = (
            mock.patch.object(iterator_module, "frontier_depth", _unbounded_depth)
            if os.environ.get("HYPERLOADER_FRONTIER_MUTATION") == "ignore-ceiling"
            else nullcontext()
        )
        try:
            if [int(value.item()) for value in loader] != list(range(64)):
                raise RuntimeError("warm frontier delivery changed sample order")
            with mutation:
                started = time.perf_counter_ns()
                delivered = []
                for value in loader:
                    delivered.append(int(value.item()))
                    time.sleep(0.004)
                elapsed_ns = time.perf_counter_ns() - started
            report = loader._last_frontier_report
        finally:
            loader.close()
    stall_fraction = report["wait_ns"] / elapsed_ns
    if delivered != list(range(64)):
        raise RuntimeError("measured frontier delivery changed sample order")
    if report["max_occupied"] > report["ceiling"]:
        raise RuntimeError("frontier occupancy exceeded its ceiling")
    if stall_fraction > 0.001:
        raise RuntimeError("frontier stalls exceeded the measured timing floor")
    if report["binding"] != "profile-tail":
        raise RuntimeError("frontier report did not name the profiled tail")
    arguments.metrics.write_text(
        json.dumps(
            {
                **report,
                "elapsed_ns": elapsed_ns,
                "stall_fraction_with_consumer": stall_fraction,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

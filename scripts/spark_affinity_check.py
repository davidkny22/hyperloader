"""Check caller-selected worker affinity on a leased measurement machine."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BENCHES = Path(__file__).parents[1] / "benches"
sys.path.insert(0, str(BENCHES))

from dominance_feeders import native_thread_affinity  # noqa: E402


def main() -> None:
    """Require inheritance of the supplied worker mask and restoration afterward."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-cpus", type=int, nargs="+", required=True)
    arguments = parser.parse_args()
    expected = set(arguments.worker_cpus)
    before = os.sched_getaffinity(0)
    with native_thread_affinity(tuple(arguments.worker_cpus)):
        during = os.sched_getaffinity(0)
        with ThreadPoolExecutor(max_workers=1) as pool:
            inherited = pool.submit(os.sched_getaffinity, 0).result()
    after = os.sched_getaffinity(0)
    if during != expected or inherited != expected or after != before:
        raise RuntimeError(
            "worker affinity did not inherit the supplied mask or restore the caller"
        )


if __name__ == "__main__":
    main()

"""Subprocess proof for extension-triggered GIL restoration telemetry."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from hyperloader import DataLoader, diagnose


class TinyDataset:
    """Provide enough work to trigger the declared thread pool detector."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> int:
        return index * index


def main() -> None:
    """Import the non-declaring extension after loader construction and report it."""
    if len(sys.argv) != 2:
        raise SystemExit("expected the built extension directory")
    query = getattr(sys, "_is_gil_enabled")
    if query():
        raise AssertionError("free-threaded subprocess started with the GIL enabled")
    loader = DataLoader(
        TinyDataset(), batch_size=None, num_workers=2, seed=401, thread_safe=True
    )
    try:
        sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
        importlib.import_module("gil_restoring")
        if not query():
            raise AssertionError("non-declaring extension did not restore the GIL")
        iterator = iter(loader)
        values = [next(iterator)]
        report = diagnose(loader).record
        events = report["gil_release"]["restore_events"]
        if events != 1:
            raise AssertionError(f"expected one restoration event, got {events!r}")
        values.extend(iterator)
        print(json.dumps({"events": events, "values": values}))
    finally:
        loader.close()


if __name__ == "__main__":
    main()

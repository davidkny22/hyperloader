"""Start Coverage.py after importing torch for Spark assurance runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import coverage
import torch

_COVERAGE: coverage.Coverage | None = None


def pytest_configure(config: Any) -> None:
    """Start branch tracing from environment-provided source roots."""
    del config
    global _COVERAGE
    source = os.environ.get("HYPERLOADER_COVERAGE_SOURCE")
    data_file = os.environ.get("HYPERLOADER_COVERAGE_FILE")
    if source is None or data_file is None:
        raise RuntimeError(
            "HYPERLOADER_COVERAGE_SOURCE and HYPERLOADER_COVERAGE_FILE are required"
        )
    torch.empty(0)
    _COVERAGE = coverage.Coverage(
        branch=True,
        data_file=str(Path(data_file).resolve()),
        source=source.split(os.pathsep),
    )
    _COVERAGE.set_option(
        "run:disable_warnings", ["already-imported", "module-not-measured"]
    )
    _COVERAGE.start()


def pytest_unconfigure(config: Any) -> None:
    """Persist branch data after the pytest session finishes."""
    del config
    global _COVERAGE
    if _COVERAGE is not None:
        _COVERAGE.stop()
        _COVERAGE.save()
        _COVERAGE = None

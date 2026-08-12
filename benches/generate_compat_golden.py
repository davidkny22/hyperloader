"""Generate one pinned Torch compatibility oracle artifact."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
from pathlib import Path

import torch

from benches.compat_golden_cases import generate_cases, supports_in_order
from benches.compat_golden_model import FORMAT, write_document


def build_document(expected_minor: str, expected_system: str) -> dict[str, object]:
    """Generate a validated artifact for the active Torch environment."""
    actual_minor = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    if actual_minor != expected_minor:
        raise RuntimeError(
            f"expected Torch {expected_minor}, found {torch.__version__}"
        )
    if platform.system().lower() != expected_system.lower():
        raise RuntimeError(
            f"expected {expected_system}, found {platform.system()}"
        )
    return {
        "format": FORMAT,
        "environment": {
            "torch": torch.__version__,
            "torch_minor": actual_minor,
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "multiprocessing_start_method": mp.get_context().get_start_method(),
            "in_order": supports_in_order(),
        },
        "cases": generate_cases(),
    }


def main() -> None:
    """Write one canonical artifact and print machine-readable provenance."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-minor", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    document = build_document(arguments.torch_minor, arguments.system)
    digest = write_document(arguments.output, document)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": digest,
                "torch": document["environment"]["torch"],
                "cases": len(document["cases"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

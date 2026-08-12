"""Reproduce one pinned Torch golden artifact from its exact environment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from benches.compat_golden_model import read_document
from benches.generate_compat_golden import build_document


def verify_artifact(path: Path) -> dict[str, object]:
    """Regenerate one stream and require exact document equality."""
    expected = read_document(path)
    environment = expected["environment"]
    actual = build_document(environment["torch_minor"], environment["system"])
    if actual["cases"] != expected["cases"]:
        location = first_difference(
            {"cases": expected["cases"]}, {"cases": actual["cases"]}
        )
        raise RuntimeError(f"torch golden reproduction differs at {location}")
    encoded = path.read_bytes()
    if not encoded.endswith(b"\n"):
        raise RuntimeError("torch golden artifact is not canonical newline-terminated JSON")
    canonical = (
        json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    if encoded != canonical:
        raise RuntimeError("torch golden artifact bytes are not canonical JSON")
    return {
        "artifact": str(path),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "torch": environment["torch"],
        "system": environment["system"],
        "cases": len(expected["cases"]),
        "epochs": sum(len(epochs) for epochs in expected["cases"].values()),
        "batches": sum(
            len(epoch)
            for epochs in expected["cases"].values()
            for epoch in epochs
        ),
        "reproduction": "bit-exact",
    }


def first_difference(expected: Any, actual: Any, path: str = "$") -> str:
    """Locate the first structural or value mismatch deterministically."""
    if type(expected) is not type(actual):
        return path
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return path
        for key in sorted(expected):
            if expected[key] != actual[key]:
                return first_difference(expected[key], actual[key], f"{path}.{key}")
        return path
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return path
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            if left != right:
                return first_difference(left, right, f"{path}[{index}]")
        return path
    return path


def main() -> None:
    """Verify one artifact and print its provenance summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(verify_artifact(arguments.artifact), sort_keys=True))


if __name__ == "__main__":
    main()

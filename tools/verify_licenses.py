"""Audit declared licenses in the locked native dependency graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ALLOWED_EXPRESSIONS = {
    "(MIT OR Apache-2.0) AND Unicode-3.0",
    "Apache-2.0 OR MIT",
    "Apache-2.0 OR MIT OR LGPL-2.1-or-later",
    "Apache-2.0 WITH LLVM-exception",
    "Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT",
    "MIT",
    "MIT OR Apache-2.0",
    "MIT OR Apache-2.0 OR LGPL-2.1-or-later",
    "MIT/Apache-2.0",
    "Unlicense OR MIT",
    "Zlib OR Apache-2.0 OR MIT",
}


def audit_metadata(metadata: dict[str, Any]) -> dict[str, int]:
    """Reject missing or unreviewed license expressions in Cargo metadata."""
    packages = metadata.get("packages")
    assert isinstance(packages, list) and packages
    expressions: set[str] = set()
    root_seen = False
    for package in packages:
        assert isinstance(package, dict)
        license_expression = package.get("license")
        assert isinstance(license_expression, str) and license_expression
        assert license_expression in ALLOWED_EXPRESSIONS, (
            package.get("name"),
            license_expression,
        )
        expressions.add(license_expression)
        if package.get("name") == "hyperloader-engine":
            assert license_expression == "MIT"
            root_seen = True
    assert root_seen
    return {"packages": len(packages), "license_expressions": len(expressions)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    arguments = parser.parse_args()
    metadata = json.loads(arguments.metadata.read_text(encoding="utf-8"))
    print(audit_metadata(metadata))


if __name__ == "__main__":
    main()

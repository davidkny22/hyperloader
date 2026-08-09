"""Run installed-wheel collation parity across pinned torch minors."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], root: Path, environment: dict[str, str]):
    return subprocess.run(
        command,
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _extract(pattern: str, output: str) -> str:
    match = re.search(pattern, output)
    if match is None:
        raise RuntimeError(f"gate output does not contain {pattern}")
    return match.group(1)


def main() -> None:
    """Run every minor, combine coverage, and prove the mismatch mutation is RED."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-root", type=Path, required=True)
    parser.add_argument("--torch-root", type=Path, action="append", required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).parents[1]
    evidence = arguments.evidence_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=False)
    coverage_file = evidence / ".coverage"
    reports = []

    for index, torch_root in enumerate(arguments.torch_root):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [
                str(arguments.install_root.resolve()),
                str(torch_root.resolve()),
                str((root / "tests").resolve()),
            ]
        )
        environment["HYPERLOADER_EXPECTED_INSTALL_ROOT"] = str(
            arguments.install_root.resolve()
        )
        environment["COVERAGE_FILE"] = str(coverage_file)
        command = [
            sys.executable,
            "-m",
            "coverage",
            "run",
        ]
        if index:
            command.append("--append")
        command.extend(
            [
                "--branch",
                "--include=*/tests/test_collate_equivalence.py",
                "-m",
                "unittest",
                "test_collate_equivalence",
                "-v",
            ]
        )
        result = _run(command, root, environment)
        output = result.stdout + result.stderr
        version = _extract(r"TORCH_VERSION=([^\r\n]+)", output)
        (evidence / f"torch-{version}.txt").write_text(output, encoding="utf-8")
        if result.returncode != 0:
            raise SystemExit(result.returncode)
        reports.append(
            {
                "error_cases": int(_extract(r"COLLATE_ERROR_CASES=(\d+)", output)),
                "torch": version,
                "value_cases": int(_extract(r"COLLATE_VALUE_CASES=(\d+)", output)),
            }
        )

    coverage = _run(
        [sys.executable, "-m", "coverage", "json", "-o", str(evidence / "coverage.json")],
        root,
        {**os.environ, "COVERAGE_FILE": str(coverage_file)},
    )
    if coverage.returncode != 0:
        raise SystemExit(coverage.returncode)

    mutation_environment = os.environ.copy()
    mutation_environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(arguments.install_root.resolve()),
            str(arguments.torch_root[0].resolve()),
            str((root / "tests").resolve()),
        ]
    )
    mutation_environment["HYPERLOADER_EXPECTED_INSTALL_ROOT"] = str(
        arguments.install_root.resolve()
    )
    mutation_environment["HYPERLOADER_COLLATE_MUTATION"] = "flip-int"
    mutation = _run(
        [sys.executable, "-m", "unittest", "test_collate_equivalence", "-v"],
        root,
        mutation_environment,
    )
    mutation_output = mutation.stdout + mutation.stderr
    (evidence / "mutation.txt").write_text(mutation_output, encoding="utf-8")
    if mutation.returncode == 0:
        raise RuntimeError("the planted collation mismatch did not make the suite fail")

    coverage_document = json.loads(
        (evidence / "coverage.json").read_text(encoding="utf-8")
    )
    summary = {
        "coverage_percent": coverage_document["totals"]["percent_covered"],
        "mutation": "RED",
        "mutation_returncode": mutation.returncode,
        "reports": reports,
    }
    (evidence / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

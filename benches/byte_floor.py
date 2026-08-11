"""Run the installed byte-floor verifier and preserve raw evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def _run(
    command: list[str], root: Path, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _test_count(output: str) -> int:
    match = re.search(r"Ran (\d+) tests?", output)
    if match is None:
        raise RuntimeError("the verifier output did not report its test count")
    return int(match.group(1))


def main() -> None:
    """Run installed per-class accounting, coverage, negative, and mutation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    arguments = parser.parse_args()

    root = Path(__file__).parents[1]
    evidence = arguments.evidence_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(arguments.install_root.resolve())
    environment["HYPERLOADER_EXPECTED_INSTALL_ROOT"] = str(
        arguments.install_root.resolve()
    )
    environment["HYPERLOADER_BYTE_FLOOR_REPORT"] = str(evidence / "classes.json")
    environment["COVERAGE_FILE"] = str(evidence / ".coverage")
    module = "tests.hyperloader.test_byte_floor"
    include = ",".join(
        [
            "*/hyperloader/api.py",
            "*/hyperloader/memory/*.py",
            "*/hyperloader/planner/structured/*.py",
            "*/hyperloader/structured/*.py",
            "*/hyperloader/tensor/*.py",
            "*/tests/hyperloader/test_byte_floor.py",
        ]
    )
    verification = _run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--branch",
            f"--include={include}",
            "-m",
            "unittest",
            module,
            "-v",
        ],
        root,
        environment,
    )
    output = verification.stdout + verification.stderr
    (evidence / "verification.txt").write_text(output, encoding="utf-8")
    if verification.returncode != 0:
        raise SystemExit(verification.returncode)

    coverage_result = _run(
        [
            sys.executable,
            "-m",
            "coverage",
            "json",
            "-o",
            str(evidence / "coverage.json"),
        ],
        root,
        environment,
    )
    (evidence / "coverage-command.txt").write_text(
        coverage_result.stdout + coverage_result.stderr, encoding="utf-8"
    )
    if coverage_result.returncode != 0:
        raise SystemExit(coverage_result.returncode)

    mutation_environment = environment.copy()
    mutation_environment.pop("HYPERLOADER_BYTE_FLOOR_REPORT", None)
    mutation_environment["HYPERLOADER_BYTE_FLOOR_MUTATION"] = "extra-payload-copy"
    mutation = _run(
        [sys.executable, "-m", "unittest", module, "-v"],
        root,
        mutation_environment,
    )
    mutation_output = mutation.stdout + mutation.stderr
    (evidence / "mutation.txt").write_text(mutation_output, encoding="utf-8")
    if mutation.returncode == 0:
        raise RuntimeError("the planted payload copy did not make the suite fail")

    coverage = json.loads((evidence / "coverage.json").read_text(encoding="utf-8"))
    classes = json.loads((evidence / "classes.json").read_text(encoding="utf-8"))
    summary = {
        "classes": len(classes),
        "coverage_percent": coverage["totals"]["percent_covered"],
        "maximum_gap_bytes_per_sample": max(
            float(row["bytes_beyond_irreducible_per_sample"])
            for row in classes.values()
        ),
        "mutation": "RED",
        "mutation_returncode": mutation.returncode,
        "negative": "shuffled tensor gathering is not declared as a storage view",
        "python": sys.version,
        "tests": _test_count(output),
    }
    (evidence / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

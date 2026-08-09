"""Run the installed worker-crash-arena gate and preserve raw evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TEST_MODULE = "tests.hyperloader.test_worker_crash_arena"


def run(
    command: list[str], root: Path, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run one captured gate subprocess."""
    return subprocess.run(
        command,
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_count(output: str) -> int:
    """Extract unittest's executed test count."""
    match = re.search(r"Ran (\d+) tests?", output)
    if match is None:
        raise RuntimeError("worker crash verifier did not report its test count")
    return int(match.group(1))


def main() -> None:
    """Measure coverage and prove the omitted-restart mutation is red."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    arguments = parser.parse_args()

    root = Path(__file__).parents[1]
    evidence = arguments.evidence_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(arguments.install_root.resolve())
    environment["COVERAGE_FILE"] = str(evidence / ".coverage")

    verification = run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--branch",
            "--source=hyperloader.process",
            "-m",
            "unittest",
            TEST_MODULE,
            "-v",
        ],
        root,
        environment,
    )
    (evidence / "verification.txt").write_text(
        verification.stdout + verification.stderr, encoding="utf-8"
    )
    if verification.returncode != 0:
        raise SystemExit(verification.returncode)

    coverage = run(
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
        coverage.stdout + coverage.stderr, encoding="utf-8"
    )
    if coverage.returncode != 0:
        raise SystemExit(coverage.returncode)

    mutation_environment = environment.copy()
    mutation_environment["HYPERLOADER_WORKER_CRASH_MUTATION"] = "omit-restart"
    mutation = run(
        [sys.executable, "-m", "unittest", TEST_MODULE, "-v"],
        root,
        mutation_environment,
    )
    mutation_output = mutation.stdout + mutation.stderr
    (evidence / "mutation.txt").write_text(mutation_output, encoding="utf-8")
    if mutation.returncode == 0:
        raise RuntimeError("the omitted-restart mutation did not make the gate fail")

    coverage_document = json.loads(
        (evidence / "coverage.json").read_text(encoding="utf-8")
    )
    summary = {
        "coverage_percent": coverage_document["totals"]["percent_covered"],
        "mutation": "RED",
        "mutation_returncode": mutation.returncode,
        "negative_assumptions": [
            "close-policy-does-not-restart",
            "healthy-worker-is-not-reclaimed",
        ],
        "python": sys.version,
        "tests": test_count(verification.stdout + verification.stderr),
    }
    (evidence / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

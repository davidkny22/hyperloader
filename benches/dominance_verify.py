"""Run installed dominance-harness assurance and preserve raw evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from benchmark_protocol.matrix import workload_names


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
    """Run installed value, protocol, coverage, negative, and mutation checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--torchvision-version")
    arguments = parser.parse_args()

    root = Path(__file__).parents[1]
    evidence = arguments.evidence_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    install_root = str(arguments.install_root.resolve())
    inherited_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((install_root, inherited_path))
        if inherited_path
        else install_root
    )
    environment["HYPERLOADER_EXPECTED_INSTALL_ROOT"] = str(
        arguments.install_root.resolve()
    )
    environment["COVERAGE_FILE"] = str(evidence / ".coverage")
    if arguments.torchvision_version is not None:
        environment["HYPERLOADER_DOMINANCE_TORCHVISION_VERSION"] = (
            arguments.torchvision_version
        )
    module = "tests.hyperloader.benchmark.test_dominance"
    include = ",".join(
        [
            "*/hyperloader/*.py",
            "*/hyperloader/memory/*.py",
            "*/hyperloader/planner/structured/*.py",
            "*/hyperloader/structured/*.py",
            "*/benches/dominance_*.py",
            "*/tests/hyperloader/benchmark/test_dominance.py",
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
    mutation_environment["HYPERLOADER_DOMINANCE_MUTATION"] = "slow-loader"
    mutation = _run(
        [
            sys.executable,
            "-m",
            "unittest",
            f"{module}.DominanceHarnessTest.test_loader_slowdown_mutation_exceeds_the_tie_margin",
            "-v",
        ],
        root,
        mutation_environment,
    )
    mutation_output = mutation.stdout + mutation.stderr
    (evidence / "mutation.txt").write_text(mutation_output, encoding="utf-8")
    if mutation.returncode == 0:
        raise RuntimeError("the planted loader slowdown did not make the suite fail")

    coverage = json.loads((evidence / "coverage.json").read_text(encoding="utf-8"))
    summary = {
        "coverage_percent": coverage["totals"]["percent_covered"],
        "mutation": "RED",
        "mutation_returncode": mutation.returncode,
        "negatives": [
            "unequal counted tuning budgets are rejected",
            "non-alternating pair order is rejected",
            "selected configurations cannot change between cells",
        ],
        "python": sys.version,
        "tests": _test_count(output),
        "workloads": len(workload_names()),
    }
    (evidence / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

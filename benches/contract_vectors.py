"""Run the contract-vector verifier and preserve its raw evidence."""

from __future__ import annotations

import argparse
import hashlib
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
    """Run coverage and mutation checks against an isolated installation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    arguments = parser.parse_args()

    root = Path(__file__).parents[1]
    evidence = arguments.evidence_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(arguments.install_root.resolve()), str((root / "tests").resolve())]
    )
    environment["HYPERLOADER_EXPECTED_INSTALL_ROOT"] = str(
        arguments.install_root.resolve()
    )
    environment["COVERAGE_FILE"] = str(evidence / ".coverage")

    include = ",".join(
        [
            "*/tests/contract_vector_harness.py",
            "*/tests/test_contract_vectors.py",
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
            "test_contract_vectors",
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
    mutation_environment["HYPERLOADER_CONTRACT_MUTATION"] = "flip-philox-word"
    mutation = _run(
        [
            sys.executable,
            "-m",
            "unittest",
            "test_contract_vectors",
            "-v",
        ],
        root,
        mutation_environment,
    )
    mutation_output = mutation.stdout + mutation.stderr
    (evidence / "mutation.txt").write_text(mutation_output, encoding="utf-8")
    if mutation.returncode == 0:
        raise RuntimeError("the planted vector mutation did not make the suite fail")

    coverage = json.loads((evidence / "coverage.json").read_text(encoding="utf-8"))
    vectors = root / "oracles" / "contract-vectors" / "vectors.json"
    document = json.loads(vectors.read_text(encoding="utf-8"))
    summary = {
        "artifact_sha256": hashlib.sha256(vectors.read_bytes()).hexdigest(),
        "coverage_percent": coverage["totals"]["percent_covered"],
        "mutation": "RED",
        "mutation_returncode": mutation.returncode,
        "permutation_vectors": len(document["permutations"]),
        "philox_vectors": len(document["philox"]["vectors"]),
        "placement_cases": len(document["placements"]),
        "python": sys.version,
        "tests": _test_count(verification.stdout + verification.stderr),
    }
    (evidence / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

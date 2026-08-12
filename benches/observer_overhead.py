"""Evaluate a raw observer-overhead report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from observer_overhead_report import evaluate_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = json.loads(arguments.input.read_text(encoding="utf-8"))
    result = evaluate_report(report)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if result["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

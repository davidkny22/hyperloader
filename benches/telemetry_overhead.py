"""Evaluate a raw telemetry overhead report and write its gate decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from telemetry_overhead_report import evaluate_report, load_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = evaluate_report(load_report(arguments.input))
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if result["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

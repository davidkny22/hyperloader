"""Validate live-training cells and emit a machine-readable point decision."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from training_eval.codec import decode_observation
from training_eval.decision import decide
from training_eval.output import write_result


def main() -> None:
    """Read paired JSONL cells, validate them, and write one JSON decision."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    observations = [
        decode_observation(json.loads(line))
        for line in arguments.observations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = decide(observations)
    write_result(
        arguments.output,
        {
            "kind": "training-throughput-decision",
            "evaluation_id": observations[0].config.evaluation_id,
            "point_id": observations[0].config.point_id,
            "config": asdict(observations[0].config),
            "decision": asdict(result),
        },
    )


if __name__ == "__main__":
    main()

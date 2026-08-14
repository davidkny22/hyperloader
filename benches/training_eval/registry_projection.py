"""Project verified training bundles into benchmark registry records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any


def project_registry_records(
    bundle: dict[str, Any], *, evidence_root: str
) -> list[dict[str, Any]]:
    """Return one verified benchmark record per reconciled training point."""
    commit = str(bundle.get("commit", ""))
    points = bundle.get("points")
    if not commit or not isinstance(points, list) or not points or not evidence_root:
        raise ValueError(
            "a verified bundle, exact commit, and evidence root are required"
        )
    records = [_project_point(point, commit, evidence_root) for point in points]
    identities = [record["id"] for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("training bundle projects duplicate registry identities")
    return records


def _project_point(
    point: dict[str, Any], commit: str, evidence_root: str
) -> dict[str, Any]:
    config = point["config"]
    environment = point["environment"]
    decision = point["decision"]
    throughput = point["throughput"]
    if environment["commit"] != commit or decision["status"] not in {"pass", "fail"}:
        raise ValueError("registry projection requires terminal commit-matched points")
    point_id = _slug(str(config["point_id"]))
    subject = _slug(str(config["subject"]))
    machine = _slug(str(environment["machine"]))
    identifier = f"training-eval-{machine}-{point_id}-{subject}-{commit[:7]}"
    terminal_reason = (
        "precision target"
        if decision["half_width_percent"]
        <= config["decision"]["max_half_width_percent"]
        else "max-pair cap"
    )
    return {
        "id": identifier,
        "claim": (
            f"{environment['machine']} {config['subject']} loader tax at "
            f"{config['point_id']}"
        ),
        "value": {
            "mean_tax_percent": decision["mean_tax_percent"],
            "throughput": throughput,
        },
        "interval": {
            "confidence": 0.95,
            "lower_percent": decision["lower_percent"],
            "upper_percent": decision["upper_percent"],
            "half_width_percent": decision["half_width_percent"],
            "pairs": decision["pairs"],
            "terminal_reason": terminal_reason,
        },
        "config": {
            **config,
            "machine_state_control": environment["machine_state_control"],
            "machine_state_cpus": environment["machine_state_cpus"],
            "machine_state_active_microseconds": environment[
                "machine_state_active_microseconds"
            ],
            "machine_state_period_microseconds": environment[
                "machine_state_period_microseconds"
            ],
            "accelerator_clock": environment["accelerator_clock"],
            "evidence": _evidence_path(point["point"], evidence_root),
        },
        "protocol": (
            "live paired mid-cell feeder swap with alternating order, equal live-step "
            "feeder tuning, matched native machine-state activity, and a terminal "
            "bootstrap interval"
        ),
        "machine": {
            key: environment[key]
            for key in (
                "machine",
                "operating_system",
                "architecture",
                "python",
                "torch",
                "accelerator",
            )
        },
        "commit": commit,
        "date": str(environment["captured_at"])[:10],
        "status": "verified",
        "history": [],
    }


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError("registry identity fields must contain letters or digits")
    return slug


def _evidence_path(point_path: object, evidence_root: str) -> str:
    path = PurePosixPath(str(point_path))
    if path.is_absolute() or len(path.parts) < 2 or ".." in path.parts:
        raise ValueError("training point evidence must be campaign-root relative")
    return PurePosixPath(evidence_root, *path.parts[1:]).as_posix()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write canonical JSON Lines without accepting a visual output path."""
    if path.suffix.lower() != ".jsonl":
        raise ValueError("registry projection output must use a .jsonl path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Project one verified result bundle into a JSONL registry shard."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    bundle = json.loads(arguments.bundle.read_text(encoding="utf-8"))
    write_jsonl(
        arguments.output,
        project_registry_records(bundle, evidence_root=arguments.evidence_root),
    )


if __name__ == "__main__":
    main()

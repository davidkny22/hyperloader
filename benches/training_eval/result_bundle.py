"""Reconcile completed training campaigns into machine-readable point records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .codec import decode_observation
from .decision import decide
from .models import TrainingObservation
from .output import write_result


class CampaignEvidenceError(ValueError):
    """A completed campaign cannot support a booked point."""


def build_bundle(campaign_roots: list[Path], *, expected_commit: str) -> dict[str, Any]:
    """Validate completed campaign outputs and return canonical point records."""
    if not campaign_roots or not expected_commit:
        raise CampaignEvidenceError("campaign roots and an exact commit are required")
    points: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for root in campaign_roots:
        summary = _read_json(root / "campaign.json")
        if summary.get("status") != "complete":
            raise CampaignEvidenceError(f"campaign is not complete: {root}")
        for item in summary.get("completed", []):
            point_path = root / Path(item["output"]).name
            point = reconcile_point(point_path, expected_commit=expected_commit)
            identity = (point["config"]["point_id"], point["config"]["subject"])
            if identity in identities:
                raise CampaignEvidenceError(f"duplicate point identity: {identity}")
            identities.add(identity)
            points.append(point)
    return {
        "kind": "training-evaluation-result-bundle",
        "commit": expected_commit,
        "campaigns": [str(path) for path in campaign_roots],
        "points": points,
    }


def reconcile_point(point: Path, *, expected_commit: str) -> dict[str, Any]:
    """Trace one point through raw observations, decision, and both guards."""
    decision_document = _read_json(point / "decision.json")
    observations = [
        decode_observation(json.loads(line))
        for line in (point / "observations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    computed = asdict(decide(observations))
    if computed != decision_document.get("decision") or computed["status"] == "collect":
        raise CampaignEvidenceError(
            f"decision is missing, nonterminal, or stale: {point}"
        )
    config = asdict(observations[0].config)
    environment = asdict(observations[0].first.environment)
    if environment["commit"] != expected_commit:
        raise CampaignEvidenceError(f"point commit does not match campaign: {point}")
    machine_state = _read_json(point / "machine-state.json")
    clock = _read_json(point.with_name(point.name + "-clock.json"))
    _validate_guards(point, environment, machine_state, clock)
    return {
        "point": str(point),
        "config": config,
        "environment": environment,
        "decision": computed,
        "throughput": _summarize_throughput(observations),
        "observations": len(observations),
        "machine_state": machine_state,
        "clock": clock,
    }


def _summarize_throughput(
    observations: list[TrainingObservation],
) -> dict[str, Any]:
    config = observations[0].config
    return {
        "aggregation": "duration-weighted mean of measured half rates",
        "epoch_duration_derivation": ("dataset_rows / measured_samples_per_second"),
        "subject": _summarize_system(observations, config.subject, config.dataset_rows),
        "reference": _summarize_system(
            observations, config.reference, config.dataset_rows
        ),
    }


def _summarize_system(
    observations: list[TrainingObservation], system: str, dataset_rows: int
) -> dict[str, Any]:
    halves = [observation.half(system) for observation in observations]
    duration = sum(half.duration_seconds for half in halves)
    if duration <= 0 or dataset_rows <= 0:
        raise CampaignEvidenceError(
            "throughput derivation requires positive duration and rows"
        )
    steps_per_second = (
        sum(half.rate_steps_per_second * half.duration_seconds for half in halves)
        / duration
    )
    samples_per_second = (
        sum(half.rate_samples_per_second * half.duration_seconds for half in halves)
        / duration
    )
    if steps_per_second <= 0 or samples_per_second <= 0:
        raise CampaignEvidenceError(
            "throughput derivation requires positive measured rates"
        )
    epoch_seconds = dataset_rows / samples_per_second
    return {
        "system": system,
        "timed_seconds": duration,
        "measured_steps_per_second": steps_per_second,
        "measured_samples_per_second": samples_per_second,
        "derived_epoch_seconds": epoch_seconds,
        "derived_epoch_minutes": epoch_seconds / 60.0,
    }


def _validate_guards(
    point: Path,
    environment: dict[str, Any],
    machine_state: dict[str, Any],
    clock: dict[str, Any],
) -> None:
    if machine_state.get("command_returncode") != 0:
        raise CampaignEvidenceError(f"machine-state command failed: {point}")
    if clock.get("command_returncode") != 0 or not clock.get("reset_stdout"):
        raise CampaignEvidenceError(f"clock guard did not close cleanly: {point}")
    if machine_state.get("control") != environment["machine_state_control"]:
        raise CampaignEvidenceError(f"machine-state control record disagrees: {point}")
    if tuple(machine_state.get("cores", ())) != tuple(
        environment["machine_state_cpus"]
    ):
        raise CampaignEvidenceError(f"machine-state CPU record disagrees: {point}")
    if (
        machine_state.get("active_microseconds")
        != environment["machine_state_active_microseconds"]
        or machine_state.get("period_microseconds")
        != environment["machine_state_period_microseconds"]
    ):
        raise CampaignEvidenceError(f"machine-state duty record disagrees: {point}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignEvidenceError(f"cannot read campaign evidence: {path}") from error
    if not isinstance(document, dict):
        raise CampaignEvidenceError(f"campaign evidence is not an object: {path}")
    return document


def main() -> None:
    """Validate campaign roots and write one canonical result bundle."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, action="append", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    write_result(
        arguments.output,
        build_bundle(arguments.campaign, expected_commit=arguments.expected_commit),
    )


if __name__ == "__main__":
    main()

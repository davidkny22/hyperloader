"""Durable controlled-variable snapshots for paired training cells."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..models import TrainingCellConfig, TrainingEnvironment
from ..output import write_result
from .registry import CONTROLLED_VARIABLE_REGISTRY, validate_registry
from .system import capture_system_state


class ControlledVariableRecorder:
    """Capture accountable state outside both timed halves."""

    def __init__(
        self,
        config: TrainingCellConfig,
        environment: TrainingEnvironment,
        feeders: Mapping[str, object],
        path: Path,
    ) -> None:
        validate_registry()
        self._config = config
        self._environment = environment
        self._feeders = feeders
        self._path = path
        self._before: dict[str, Any] | None = None

    def capture_before_collection(self) -> None:
        """Persist the warm, pre-timing state and its enforced controls."""
        if self._path.exists():
            raise FileExistsError("controlled-variable output already exists")
        self._before = self._snapshot()
        self._write(status="collecting", after=None)

    def capture_after_collection(self) -> None:
        """Persist terminal state while every feeder remains alive."""
        if self._before is None:
            raise RuntimeError("controlled variables require a pre-collection snapshot")
        self._write(status="complete", after=self._snapshot())

    def _snapshot(self) -> dict[str, Any]:
        values = capture_system_state(self._config, self._environment)
        feeder_records = {
            name: _feeder_snapshot(feeder) for name, feeder in self._feeders.items()
        }
        values["process_affinity"] = {
            "consumer": values["process_affinity"]["consumer"],
            "feeders": {
                name: _process_field(record, "affinity")
                for name, record in feeder_records.items()
            },
        }
        values["process_thread_counts"] = {
            "consumer": values["process_thread_counts"]["consumer"],
            "feeders": {
                name: _process_field(record, "os_thread_count")
                for name, record in feeder_records.items()
            },
        }
        values["thread_environment"] = {
            "consumer": values["thread_environment"],
            "feeders": {
                name: {
                    "processes": _process_field(record, "environment"),
                    "worker_boot": [
                        item["environment"] for item in record.get("worker_boot", [])
                    ],
                }
                for name, record in feeder_records.items()
            },
        }
        values["torch_threading"] = {
            "consumer": values["torch_threading"],
            "feeders": {
                name: [
                    {
                        "pid": item["pid"],
                        "worker_id": item["worker_id"],
                        "torch_inter_op_threads": item["torch_inter_op_threads"],
                        "torch_intra_op_threads": item["torch_intra_op_threads"],
                    }
                    for item in record.get("worker_boot", [])
                ]
                for name, record in feeder_records.items()
            },
        }
        _validate_snapshot(values, self._environment)
        return {"captured_at": datetime.now(UTC).isoformat(), "values": values}

    def _write(self, *, status: str, after: dict[str, Any] | None) -> None:
        write_result(
            self._path,
            {
                "kind": "training-controlled-variables",
                "schema_version": 1,
                "status": status,
                "registry": [asdict(entry) for entry in CONTROLLED_VARIABLE_REGISTRY],
                "before_collection": self._before,
                "after_collection": after,
            },
        )


def validate_control_document(document: dict[str, Any]) -> None:
    """Reject incomplete or structurally stale controlled-variable evidence."""
    validate_registry()
    if document.get("kind") != "training-controlled-variables":
        raise ValueError("controlled-variable evidence has the wrong kind")
    if document.get("schema_version") != 1 or document.get("status") != "complete":
        raise ValueError("controlled-variable evidence is not terminal")
    expected_registry = [asdict(entry) for entry in CONTROLLED_VARIABLE_REGISTRY]
    if document.get("registry") != expected_registry:
        raise ValueError("controlled-variable registry does not match the harness")
    for key in ("before_collection", "after_collection"):
        snapshot = document.get(key)
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("values"), dict):
            raise TypeError(f"controlled-variable evidence lacks {key}")
        _validate_names(snapshot["values"])


def _feeder_snapshot(feeder: object) -> dict[str, Any]:
    capture = getattr(feeder, "control_snapshot", None)
    if capture is None:
        return {
            "system": getattr(feeder, "system", type(feeder).__name__),
            "execution": "consumer-process",
        }
    result = capture()
    if not isinstance(result, dict):
        raise TypeError("feeder control snapshot must be an object")
    return result


def _validate_snapshot(
    values: dict[str, Any], environment: TrainingEnvironment
) -> None:
    _validate_names(values)
    if environment.interactive_load:
        raise ValueError("paired training cells require interactive load to be absent")
    if not environment.thermal_steady:
        raise ValueError("paired training cells require thermal steady state")
    if not environment.lease_kind or not environment.lease_token:
        raise ValueError("paired training cells require a resolved machine lease")
    observed = values["governor_and_power"]["observed_cpu_governors"]
    if environment.cpu_governor != "uncontrolled" and isinstance(observed, dict) and any(
        governor != environment.cpu_governor for governor in observed.values()
    ):
        raise ValueError("observed CPU governor disagrees with the cell declaration")


def _validate_names(values: dict[str, Any]) -> None:
    expected = {entry.name for entry in CONTROLLED_VARIABLE_REGISTRY}
    if set(values) != expected:
        missing = sorted(expected - set(values))
        extra = sorted(set(values) - expected)
        raise ValueError(f"controlled-variable names differ: missing={missing}, extra={extra}")


def _process_field(record: dict[str, Any], field: str) -> list[dict[str, Any]]:
    return [
        {"pid": process["pid"], field: process[field]}
        for process in record.get("processes", [])
    ]

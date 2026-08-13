"""Behavioral assurance for paired-cell controlled-variable evidence."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from benches.training_eval.controls.capture import validate_control_document
from benches.training_eval.controls.registry import (
    CONTROLLED_VARIABLE_REGISTRY,
    validate_registry,
)
from benches.training_eval.controls.worker_probe import WorkerEnvironmentProbe


def test_registry_names_are_unique_and_classifications_are_closed() -> None:
    validate_registry()
    assert len(CONTROLLED_VARIABLE_REGISTRY) == len(
        {entry.name for entry in CONTROLLED_VARIABLE_REGISTRY}
    )


def test_worker_probe_records_the_enforced_torch_thread_count(tmp_path: Path) -> None:
    probe = WorkerEnvironmentProbe(str(tmp_path))
    with patch("torch.get_num_threads", return_value=1):
        probe(3)

    records = list(tmp_path.glob("worker-*.json"))
    assert len(records) == 1
    document = json.loads(records[0].read_text(encoding="utf-8"))
    assert document["worker_id"] == 3
    assert document["torch_intra_op_threads"] == 1


def test_worker_probe_rejects_oversubscribed_boot(tmp_path: Path) -> None:
    probe = WorkerEnvironmentProbe(str(tmp_path))
    with (
        patch("torch.get_num_threads", return_value=2),
        pytest.raises(RuntimeError, match="one Torch intra-op thread"),
    ):
        probe(0)
    assert not list(tmp_path.iterdir())


def test_terminal_control_document_requires_every_registered_family() -> None:
    values = {entry.name: {} for entry in CONTROLLED_VARIABLE_REGISTRY}
    document = {
        "kind": "training-controlled-variables",
        "schema_version": 1,
        "status": "complete",
        "registry": [
            {
                "name": entry.name,
                "classification": entry.classification,
                "rationale": entry.rationale,
            }
            for entry in CONTROLLED_VARIABLE_REGISTRY
        ],
        "before_collection": {"captured_at": "before", "values": values},
        "after_collection": {"captured_at": "after", "values": values},
    }
    validate_control_document(document)
    document["after_collection"]["values"].pop("thermal_state")
    with pytest.raises(ValueError, match="names differ"):
        validate_control_document(document)

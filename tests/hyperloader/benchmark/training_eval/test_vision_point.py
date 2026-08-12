"""Image-folder point collection behavior."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from benches.training_eval.image_folder import TrainingImageFolder
from benches.training_eval.models import (
    DecisionRule,
    TrainingCellConfig,
    TrainingEnvironment,
)
from benches.training_eval.vision_point import collect_vision_point


def test_image_identity_tracks_source_bytes(tmp_path: Path) -> None:
    root = _image_root(tmp_path)
    first = TrainingImageFolder(root, resolution=16, seed=7)
    identity = first.identity_for_rows(2)
    assert torch.equal(first[0][0], first[0][0])
    changed_seed = TrainingImageFolder(root, resolution=16, seed=8)
    assert changed_seed.identity_for_rows(2) != identity
    Image.new("RGB", (20, 20), (99, 1, 2)).save(root / "class-a" / "0.png")
    changed = TrainingImageFolder(root, resolution=16, seed=7)
    assert first.class_count == 2
    assert changed.identity_for_rows(2) != identity


def test_vision_point_uses_the_shared_live_protocol(tmp_path: Path) -> None:
    dataset = TrainingImageFolder(_image_root(tmp_path), resolution=16, seed=0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 2))
    parameters = sum(value.numel() for value in model.parameters())
    observations = tmp_path / "observations.jsonl"
    decision = tmp_path / "decision.json"
    result = collect_vision_point(
        _config(dataset, parameters),
        _environment(),
        dataset=dataset,
        model=model,
        device=torch.device("cpu"),
        pin_memory=False,
        observations_path=observations,
        decision_path=decision,
    )
    document = json.loads(decision.read_text(encoding="utf-8"))
    assert result.status == "pass"
    assert len(observations.read_text(encoding="utf-8").splitlines()) == 1
    assert document["config"]["input_resolution"] == 16
    assert document["config"]["dataset_identity"] == dataset.identity_for_rows(2)


def _image_root(tmp_path: Path) -> Path:
    root = tmp_path / "images"
    for class_name, offset in (("class-a", 10), ("class-b", 20)):
        directory = root / class_name
        directory.mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (20, 20), (offset + index, 2, 3)).save(
                directory / f"{index}.png"
            )
    return root


def _config(dataset: TrainingImageFolder, parameters: int) -> TrainingCellConfig:
    return TrainingCellConfig(
        evaluation_id="test-evaluation",
        point_id="resnet18-image-folder-finetuning",
        comparison_kind="loader-tax",
        subject="torch",
        reference="counterfactual",
        workload_family="vision-finetuning",
        data_class="image-folder-standard-augmentation",
        batch_size=2,
        sequence_length=None,
        input_resolution=16,
        model_width=None,
        model_depth=1,
        attention_heads=None,
        precision="float32",
        optimizer="AdamW(lr=0.0003)",
        learning_rate=0.0003,
        delivery="pageable",
        device="cpu",
        model_name="test classifier",
        model_parameters=parameters,
        dataset_rows=2,
        dataset_identity=dataset.identity_for_rows(2),
        seed=0,
        resident_batches=1,
        warmup_steps=1,
        subject_workers=0,
        reference_workers=0,
        subject_prefetch=1,
        reference_prefetch=1,
        half_seconds=0.001,
        tuning_trials=1,
        tuning_seconds=1.0,
        tuning_knobs=("workers", "prefetch"),
        decision=DecisionRule(
            threshold_percent=100.0,
            min_pairs=1,
            max_pairs=1,
            max_half_width_percent=100.0,
            bootstrap_draws=100,
        ),
    )


def _environment() -> TrainingEnvironment:
    return TrainingEnvironment(
        captured_at="runtime-captured",
        machine="runtime-machine",
        operating_system="runtime-system",
        architecture="runtime-architecture",
        python="runtime-python",
        torch="runtime-torch",
        accelerator="runtime-accelerator",
        accelerator_clock="runtime-clock",
        memory_clock="runtime-memory-clock",
        cpu_governor="runtime-governor",
        power_profile="runtime-profile",
        plugged_in=None,
        thermal_steady=True,
        interactive_load=False,
        commit="runtime-commit",
        lease_kind="SPARK-LOCK",
        lease_token="deadbeef",
        ambient_probe_id="runtime-probe",
    )

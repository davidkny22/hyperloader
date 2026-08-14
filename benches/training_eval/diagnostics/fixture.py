"""Recorded-config reconstruction for training diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

from ..codec import decode_config
from ..dial import default_dial
from ..image_folder import TrainingImageFolder
from ..token_point import _build_feeders as build_token_feeders
from ..token_point import _resident_batches as resident_token_batches
from ..token_source import PretokenizedRows
from ..training_step import TransformerStepRunner
from ..transformer import DialTransformer
from ..vision_model import build_resnet18
from ..vision_point import _build_feeders as build_vision_feeders
from ..vision_point import _resident_batches as resident_vision_batches
from ..vision_step import VisionStepRunner


@dataclass
class DiagnosticFixture:
    """Own the feeders and step runner reconstructed from one terminal point."""

    config: Any
    feeders: dict[str, Any]
    runner: TransformerStepRunner | VisionStepRunner

    def close(self) -> None:
        """Close every loader resource."""
        for feeder in self.feeders.values():
            close = getattr(feeder, "close", None)
            if close is not None:
                close()


def build_fixture(
    decision_path: Path,
    *,
    output: Path,
    image_root: Path | None,
) -> DiagnosticFixture:
    """Reconstruct one diagnostic workload from its accepted decision config."""
    document = json.loads(decision_path.read_text(encoding="utf-8"))
    config = decode_config(document["config"])
    torch.manual_seed(config.seed)
    if config.workload_family == "transformer-dial":
        point = next(item for item in default_dial() if item.point_id == config.point_id)
        dataset = PretokenizedRows(
            rows=config.dataset_rows,
            sequence_length=point.sequence_length,
            vocabulary_size=point.vocabulary_size,
            seed=config.seed,
        )
        resident = resident_token_batches(
            dataset, batch_size=config.batch_size, pin_memory=True
        )
        feeders = build_token_feeders(
            config,
            dataset,
            resident,
            pin_memory=True,
            worker_environment_dir=output / "worker-environment",
        )
        runner = TransformerStepRunner(
            DialTransformer(point),
            device=torch.device(config.device),
            precision=config.precision,
            learning_rate=config.learning_rate,
            non_blocking=True,
        )
    elif config.workload_family == "vision-finetuning":
        if image_root is None:
            raise ValueError("vision diagnostics require an image root")
        dataset = TrainingImageFolder(
            image_root, resolution=int(config.input_resolution), seed=config.seed
        )
        subset = Subset(dataset, range(config.dataset_rows))
        resident = resident_vision_batches(
            subset, batch_size=config.batch_size, pin_memory=True
        )
        feeders = build_vision_feeders(
            config,
            subset,
            resident,
            pin_memory=True,
            worker_environment_dir=output / "worker-environment",
        )
        runner = VisionStepRunner(
            build_resnet18(classes=dataset.class_count),
            device=torch.device(config.device),
            precision=config.precision,
            learning_rate=config.learning_rate,
            non_blocking=True,
        )
    else:
        raise ValueError(f"unsupported diagnostic workload {config.workload_family!r}")
    return DiagnosticFixture(config, feeders, runner)

"""Collection of one live image-folder training comparison point."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Subset

from .decision import TrainingDecision
from .image_batches import ImageBatch
from .image_folder import TrainingImageFolder, collate_image_batch
from .live_cell import BatchFeeder
from .models import TrainingCellConfig, TrainingEnvironment
from .point_collection import collect_point
from .public_feeders import build_public_feeder
from .vision_step import VisionStepRunner


class ResidentImageFeeder:
    """Cycle through pre-materialized image batches without loader work."""

    def __init__(self, system: str, batches: tuple[ImageBatch, ...]) -> None:
        if not system or not batches:
            raise ValueError("resident image feeders require a name and batch bank")
        self.system = system
        self._batches = batches
        self._index = 0

    def next_batch(self) -> ImageBatch:
        """Return the next resident image batch."""
        batch = self._batches[self._index]
        self._index = (self._index + 1) % len(self._batches)
        return batch


def collect_vision_point(
    config: TrainingCellConfig,
    environment: TrainingEnvironment,
    *,
    dataset: TrainingImageFolder,
    model: nn.Module,
    device: torch.device,
    pin_memory: bool,
    observations_path: Path,
    decision_path: Path,
) -> TrainingDecision:
    """Collect one ResNet image-folder point through the shared live protocol."""
    if config.input_resolution is None:
        raise ValueError("vision points require an input resolution")
    if config.dataset_rows != config.resident_batches * config.batch_size:
        raise ValueError("vision dataset rows must equal the resident batch bank")
    if dataset.identity_for_rows(config.dataset_rows) != config.dataset_identity:
        raise ValueError("image source does not match the recorded dataset identity")
    parameters = sum(
        value.numel() for value in model.parameters() if value.requires_grad
    )
    if parameters != config.model_parameters:
        raise ValueError("vision model does not match the recorded parameter count")
    subset = Subset(dataset, range(config.dataset_rows))
    resident = _resident_batches(
        subset,
        batch_size=config.batch_size,
        pin_memory=pin_memory,
    )
    feeders = _build_feeders(
        config,
        subset,
        resident,
        pin_memory=pin_memory,
        worker_environment_dir=observations_path.parent / "worker-environment",
    )
    runner = VisionStepRunner(
        model,
        device=device,
        precision=config.precision,
        learning_rate=config.learning_rate,
        non_blocking=pin_memory,
    )
    return collect_point(
        config,
        environment,
        feeders=feeders,
        runner=runner,
        warmup_steps=config.warmup_steps,
        observations_path=observations_path,
        decision_path=decision_path,
    )


def _resident_batches(
    dataset: Subset[tuple[torch.Tensor, int, str]],
    *,
    batch_size: int,
    pin_memory: bool,
) -> tuple[ImageBatch, ...]:
    batches = []
    for start in range(0, len(dataset), batch_size):
        batch = collate_image_batch(
            [dataset[index] for index in range(start, start + batch_size)]
        )
        if pin_memory:
            batch = ImageBatch(
                batch.images.pin_memory(), batch.labels.pin_memory(), batch.digest
            )
        batches.append(batch)
    return tuple(batches)


def _build_feeders(
    config: TrainingCellConfig,
    dataset: Subset[tuple[torch.Tensor, int, str]],
    resident: tuple[ImageBatch, ...],
    *,
    pin_memory: bool,
    worker_environment_dir: Path | None = None,
) -> dict[str, BatchFeeder]:
    feeders: dict[str, BatchFeeder] = {
        config.reference: ResidentImageFeeder(config.reference, resident)
    }
    feeders[config.subject] = build_public_feeder(
        config.subject,
        dataset,
        batch_size=config.batch_size,
        workers=config.subject_workers,
        prefetch=config.subject_prefetch,
        collate=collate_image_batch,
        pin_memory=pin_memory,
        worker_environment_dir=worker_environment_dir,
    )
    return feeders

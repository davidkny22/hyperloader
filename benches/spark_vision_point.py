"""Collect one portable Spark ResNet image-folder comparison point."""

from __future__ import annotations

import argparse
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from benches.training_eval.image_folder import TrainingImageFolder
from benches.training_eval.machine_state import (
    add_machine_state_arguments,
    machine_state_environment_fields,
)
from benches.training_eval.models import (
    DecisionRule,
    TrainingCellConfig,
    TrainingEnvironment,
)
from benches.training_eval.vision_model import build_resnet18
from benches.training_eval.vision_point import collect_vision_point
from benches.training_eval.vision_tuning import tune_vision_system
from benches.training_eval.tuning import parse_tuning_candidates


def main() -> None:
    """Parse runtime facts and collect one ResNet-18 loader comparison."""
    arguments = _parser().parse_args()
    torch.manual_seed(arguments.seed)
    dataset = TrainingImageFolder(
        arguments.image_root, resolution=arguments.resolution, seed=arguments.seed
    )
    rows = arguments.bank_batches * arguments.batch_size
    if rows > len(dataset):
        raise ValueError("image-folder source is smaller than the resident bank")
    if arguments.output.exists() and any(arguments.output.iterdir()):
        raise FileExistsError("training point output directory is not empty")
    arguments.output.mkdir(parents=True, exist_ok=True)
    candidates = parse_tuning_candidates(arguments.tuning_candidate)
    if not candidates:
        raise ValueError("claim-bearing points require tuning candidates")

    def model_factory():
        return build_resnet18(classes=dataset.class_count)

    selected = tune_vision_system(
        arguments.subject,
        dataset=dataset,
        rows=rows,
        batch_size=arguments.batch_size,
        model_factory=model_factory,
        candidates=candidates,
        device=torch.device(arguments.device),
        precision=arguments.precision,
        learning_rate=0.0003,
        pin_memory=arguments.pin_memory,
        seed=arguments.seed,
        seconds_per_trial=arguments.tuning_seconds,
        warmup_steps=arguments.tuning_warmup_steps,
        output=arguments.output / "tuning.json",
    )
    torch.manual_seed(arguments.seed)
    model = model_factory()
    config = TrainingCellConfig(
        evaluation_id=arguments.evaluation_id,
        point_id="resnet18-image-folder-finetuning",
        comparison_kind="loader-tax",
        subject=arguments.subject,
        reference="counterfactual",
        workload_family="vision-finetuning",
        data_class="image-folder-standard-augmentation",
        batch_size=arguments.batch_size,
        sequence_length=None,
        input_resolution=arguments.resolution,
        model_width=None,
        model_depth=18,
        attention_heads=None,
        precision=arguments.precision,
        optimizer="AdamW(lr=0.0003)",
        learning_rate=0.0003,
        delivery="pinned" if arguments.pin_memory else "pageable",
        device=arguments.device,
        model_name="ResNet-18",
        model_parameters=sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
        dataset_rows=rows,
        dataset_identity=dataset.identity_for_rows(rows),
        seed=arguments.seed,
        resident_batches=arguments.bank_batches,
        warmup_steps=arguments.warmup_steps,
        subject_workers=selected.workers,
        reference_workers=0,
        subject_prefetch=selected.prefetch,
        reference_prefetch=1,
        half_seconds=arguments.half_seconds,
        tuning_trials=len(candidates),
        tuning_seconds=len(candidates) * arguments.tuning_seconds,
        tuning_knobs=("workers", "prefetch"),
        decision=DecisionRule(
            threshold_percent=arguments.threshold_percent,
            min_pairs=arguments.min_pairs,
            max_pairs=arguments.max_pairs,
            max_half_width_percent=arguments.max_half_width_percent,
            bootstrap_draws=arguments.bootstrap_draws,
            bootstrap_seed=arguments.bootstrap_seed,
        ),
    )
    environment = TrainingEnvironment(
        captured_at=datetime.now(UTC).isoformat(),
        machine=arguments.machine,
        operating_system=platform.system(),
        architecture=platform.machine(),
        python=platform.python_version(),
        torch=torch.__version__,
        accelerator=torch.cuda.get_device_name(arguments.device),
        accelerator_clock=arguments.accelerator_clock,
        memory_clock=arguments.memory_clock,
        cpu_governor=arguments.cpu_governor,
        power_profile=arguments.power_profile,
        plugged_in=None,
        thermal_steady=arguments.thermal_steady,
        interactive_load=False,
        commit=arguments.commit,
        lease_kind="SPARK-LOCK",
        lease_token=arguments.lease_token,
        ambient_probe_id=arguments.ambient_probe_id,
        **machine_state_environment_fields(arguments),
    )
    result = collect_vision_point(
        config,
        environment,
        dataset=dataset,
        model=model,
        device=torch.device(arguments.device),
        pin_memory=arguments.pin_memory,
        observations_path=arguments.output / "observations.jsonl",
        decision_path=arguments.output / "decision.json",
    )
    print(asdict(result))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject", choices=("torch", "hyperloader", "spdl"), required=True
    )
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--lease-token", required=True)
    parser.add_argument("--accelerator-clock", required=True)
    parser.add_argument("--memory-clock", required=True)
    parser.add_argument("--cpu-governor", required=True)
    parser.add_argument("--power-profile", required=True)
    parser.add_argument("--ambient-probe-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--half-seconds", type=float, default=45.0)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--bank-batches", type=int, default=64)
    parser.add_argument("--min-pairs", type=int, default=10)
    parser.add_argument("--max-pairs", type=int, default=40)
    parser.add_argument("--max-half-width-percent", type=float, default=0.15)
    parser.add_argument("--threshold-percent", type=float, default=2.0)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--tuning-candidate", action="append", default=[])
    parser.add_argument("--tuning-seconds", type=float, default=2.0)
    parser.add_argument("--tuning-warmup-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--thermal-steady", action=argparse.BooleanOptionalAction, default=True
    )
    add_machine_state_arguments(parser)
    return parser


if __name__ == "__main__":
    main()

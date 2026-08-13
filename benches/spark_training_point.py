"""Collect one portable Spark token-training comparison point."""

from __future__ import annotations

import argparse
import platform
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn

from benches.training_eval.dial import TransformerDialPoint, default_dial
from benches.training_eval.gpt import GPT2_124M, GPT2_355M, GptLanguageModel
from benches.training_eval.machine_state import (
    add_machine_state_arguments,
    machine_state_environment_fields,
)
from benches.training_eval.models import (
    DecisionRule,
    TrainingCellConfig,
    TrainingEnvironment,
)
from benches.training_eval.token_point import collect_token_point
from benches.training_eval.token_source import PretokenizedRows
from benches.training_eval.token_tuning import tune_token_system
from benches.training_eval.transformer import DialTransformer
from benches.training_eval.tuning import parse_tuning_candidates


def main() -> None:
    """Parse runtime facts and collect one null, dial, or GPT comparison."""
    arguments = _parser().parse_args()
    point, model_factory = _point(arguments)
    if arguments.output.exists() and any(arguments.output.iterdir()):
        raise FileExistsError("training point output directory is not empty")
    arguments.output.mkdir(parents=True, exist_ok=True)
    subject = "null-b" if arguments.kind == "null" else arguments.subject
    reference = "null-a" if arguments.kind == "null" else "counterfactual"
    mode = "absolute" if arguments.kind == "null" else "upper"
    rows = arguments.bank_batches * int(point["batch_size"])
    dataset = PretokenizedRows(
        rows=rows,
        sequence_length=int(point["sequence_length"]),
        vocabulary_size=int(point["vocabulary_size"]),
        seed=arguments.seed,
    )
    candidates = parse_tuning_candidates(arguments.tuning_candidate)
    if arguments.kind == "null":
        if candidates:
            raise ValueError("null points do not tune feeder controls")
        subject_workers = 0
        subject_prefetch = 1
    else:
        if not candidates:
            raise ValueError("claim-bearing points require tuning candidates")
        selected = tune_token_system(
            subject,
            dataset=dataset,
            batch_size=int(point["batch_size"]),
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
        subject_workers = selected.workers
        subject_prefetch = selected.prefetch
    config = TrainingCellConfig(
        evaluation_id=arguments.evaluation_id,
        point_id=point["point_id"],
        comparison_kind="null" if arguments.kind == "null" else "loader-tax",
        subject=subject,
        reference=reference,
        workload_family=point["family"],
        data_class="pretokenized-text",
        batch_size=point["batch_size"],
        sequence_length=point["sequence_length"],
        input_resolution=None,
        model_width=point["width"],
        model_depth=point["depth"],
        attention_heads=point["heads"],
        precision=arguments.precision,
        optimizer="AdamW(lr=0.0003)",
        learning_rate=0.0003,
        delivery="pinned" if arguments.pin_memory else "pageable",
        device=arguments.device,
        model_name=str(point["model_name"]),
        model_parameters=int(point["model_parameters"]),
        dataset_rows=rows,
        dataset_identity=dataset.identity,
        seed=arguments.seed,
        resident_batches=arguments.bank_batches,
        warmup_steps=arguments.warmup_steps,
        subject_workers=subject_workers,
        reference_workers=0,
        subject_prefetch=subject_prefetch,
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
            mode=mode,
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
    result = collect_token_point(
        config,
        environment,
        model_factory=model_factory,
        device=torch.device(arguments.device),
        seed=arguments.seed,
        bank_batches=arguments.bank_batches,
        warmup_steps=arguments.warmup_steps,
        pin_memory=arguments.pin_memory,
        observations_path=arguments.output / "observations.jsonl",
        decision_path=arguments.output / "decision.json",
    )
    print(asdict(result))


def _point(
    arguments: argparse.Namespace,
) -> tuple[dict[str, int | str], Callable[[], nn.Module]]:
    if arguments.kind in {"null", "dial"}:
        point = default_dial()[arguments.dial_index - 1]
        values = {
            "point_id": "null" if arguments.kind == "null" else point.point_id,
            "family": "transformer-dial",
            "batch_size": point.batch_size,
            "sequence_length": point.sequence_length,
            "width": point.width,
            "depth": point.depth,
            "heads": point.attention_heads,
            "model_name": "synthetic transformer dial",
            "model_parameters": _dial_parameter_count(point),
            "vocabulary_size": point.vocabulary_size,
        }
        return values, lambda: DialTransformer(point)
    gpt = GPT2_124M if arguments.kind == "gpt2-124m" else GPT2_355M
    batch_size = 8 if arguments.kind == "gpt2-124m" else 2
    values = {
        "point_id": f"{arguments.kind}-pretraining",
        "family": "gpt-pretraining",
        "batch_size": batch_size,
        "sequence_length": 256,
        "width": gpt.width,
        "depth": gpt.depth,
        "heads": gpt.attention_heads,
        "model_name": gpt.name,
        "model_parameters": gpt.parameter_count(),
        "vocabulary_size": gpt.vocabulary_size,
    }
    return values, lambda: GptLanguageModel(gpt)


def _dial_parameter_count(point: TransformerDialPoint) -> int:
    width = point.width
    vocabulary = point.vocabulary_size
    per_layer = 12 * width * width + 13 * width
    return 2 * vocabulary * width + vocabulary + point.depth * per_layer + 2 * width


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind", choices=("null", "dial", "gpt2-124m", "gpt2-355m"), required=True
    )
    parser.add_argument(
        "--subject", choices=("torch", "hyperloader", "spdl"), default="hyperloader"
    )
    parser.add_argument("--dial-index", type=int, choices=range(1, 9), default=1)
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
    parser.add_argument("--half-seconds", type=float, default=45.0)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--bank-batches", type=int, default=64)
    parser.add_argument("--min-pairs", type=int, default=10)
    parser.add_argument("--max-pairs", type=int, default=40)
    parser.add_argument("--max-half-width-percent", type=float, default=0.15)
    parser.add_argument("--threshold-percent", type=float, required=True)
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

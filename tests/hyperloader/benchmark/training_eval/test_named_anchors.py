"""Recognizable GPT and vision anchor behavior tests."""

from __future__ import annotations

import hashlib
import math

import torch

from benches.training_eval import default_named_anchors
from benches.training_eval.feeders import TokenBatch
from benches.training_eval.gpt import GPT2_124M, GPT2_355M, GptConfig, GptLanguageModel
from benches.training_eval.image_batches import ImageBatch
from benches.training_eval.image_folder import collate_image_batch
from benches.training_eval.training_step import TransformerStepRunner
from benches.training_eval.vision_model import build_resnet18
from benches.training_eval.vision_step import VisionStepRunner


def test_named_anchors_cover_two_gpt_sizes_and_resnet() -> None:
    anchors = default_named_anchors()
    assert [anchor.family for anchor in anchors].count("gpt-pretraining") == 2
    assert [anchor.family for anchor in anchors].count("vision-finetuning") == 1
    assert all(anchor.references[-1] == "spdl" for anchor in anchors)
    assert 120_000_000 <= GPT2_124M.parameter_count() < 130_000_000
    assert 350_000_000 <= GPT2_355M.parameter_count() < 360_000_000


def test_tiny_gpt_executes_a_real_causal_optimizer_step() -> None:
    torch.manual_seed(11)
    config = GptConfig("tiny-gpt", 8, 1, 2, vocabulary_size=16, max_positions=8)
    model = GptLanguageModel(config)
    runner = TransformerStepRunner(
        model,
        device=torch.device("cpu"),
        precision="float32",
        learning_rate=0.01,
        non_blocking=False,
    )
    tokens = torch.randint(0, config.vocabulary_size, (2, 5))
    batch = TokenBatch(tokens, hashlib.sha256(tokens.numpy().tobytes()).hexdigest())
    assert math.isfinite(runner.finish(runner.step(batch)))


def test_resnet_finetune_step_and_image_collation_are_executable() -> None:
    torch.manual_seed(13)
    rows = [
        (
            torch.rand((3, 32, 32)),
            index,
            hashlib.sha256(bytes([index])).hexdigest(),
        )
        for index in range(2)
    ]
    batch = collate_image_batch(rows)
    assert isinstance(batch, ImageBatch)
    runner = VisionStepRunner(
        build_resnet18(classes=2),
        device=torch.device("cpu"),
        precision="float32",
        learning_rate=0.01,
        non_blocking=False,
    )
    assert math.isfinite(runner.finish(runner.step(batch)))

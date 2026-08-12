"""Real transformer dial execution tests."""

from __future__ import annotations

import hashlib
import math

import torch

from benches.training_eval.dial import TransformerDialPoint
from benches.training_eval.feeders import TokenBatch
from benches.training_eval.training_step import TransformerStepRunner
from benches.training_eval.transformer import DialTransformer


def test_cpu_step_runs_forward_backward_and_optimizer() -> None:
    torch.manual_seed(7)
    point = TransformerDialPoint("tiny", 8, 1, 2, 4, 2, 16)
    model = DialTransformer(point)
    before = model.embedding.weight.detach().clone()
    runner = TransformerStepRunner(
        model,
        device=torch.device("cpu"),
        precision="float32",
        learning_rate=0.01,
        non_blocking=False,
    )
    tokens = torch.randint(0, point.vocabulary_size, (2, 4))
    batch = TokenBatch(tokens, hashlib.sha256(tokens.numpy().tobytes()).hexdigest())
    loss = runner.finish(runner.step(batch))
    assert math.isfinite(loss)
    assert not torch.equal(before, model.embedding.weight.detach())

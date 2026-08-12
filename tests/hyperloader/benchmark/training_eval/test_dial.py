"""Transformer step-time dial behavior tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from benches.training_eval import TransformerDialPoint, default_dial, validate_dial


def test_default_dial_has_eight_valid_unique_points() -> None:
    points = default_dial()
    assert len(points) == 8
    assert len({point.point_id for point in points}) == 8
    assert all(point.width % point.attention_heads == 0 for point in points)


def test_dial_rejects_wrong_count_and_invalid_attention_width() -> None:
    point = TransformerDialPoint("point", 8, 1, 2, 4, 2, 16)
    with pytest.raises(ValueError, match="eight to ten"):
        validate_dial((point,) * 7)
    invalid = replace(point, width=9)
    points = tuple(replace(point, point_id=f"point-{index}") for index in range(7))
    with pytest.raises(ValueError, match="divisible"):
        validate_dial((*points, replace(invalid, point_id="invalid")))

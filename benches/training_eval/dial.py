"""Configurable transformer shapes for the live step-time sweep."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerDialPoint:
    """One synthetic-architecture point with real transformer compute."""

    point_id: str
    width: int
    depth: int
    attention_heads: int
    sequence_length: int
    batch_size: int
    vocabulary_size: int

    def validate(self) -> None:
        """Reject shapes that cannot form a transformer language-model step."""
        dimensions = (
            self.width,
            self.depth,
            self.attention_heads,
            self.sequence_length,
            self.batch_size,
            self.vocabulary_size,
        )
        if not self.point_id or any(value <= 0 for value in dimensions):
            raise ValueError("transformer dial dimensions and point id must be positive")
        if self.width % self.attention_heads:
            raise ValueError("model width must be divisible by the attention-head count")
        if self.sequence_length < 2:
            raise ValueError("next-token training requires at least two tokens")


def validate_dial(points: tuple[TransformerDialPoint, ...]) -> None:
    """Require the preregistered point count and unique ordered identities."""
    if not 8 <= len(points) <= 10:
        raise ValueError("the transformer dial requires eight to ten points")
    for point in points:
        point.validate()
    identities = [point.point_id for point in points]
    if len(set(identities)) != len(identities):
        raise ValueError("transformer dial point ids must be unique")


def default_dial() -> tuple[TransformerDialPoint, ...]:
    """Return eight bounded points for calibration on both evaluation machines."""
    points = (
        TransformerDialPoint("dial-01", 64, 1, 4, 64, 16, 2048),
        TransformerDialPoint("dial-02", 96, 2, 4, 64, 16, 2048),
        TransformerDialPoint("dial-03", 128, 2, 4, 96, 16, 2048),
        TransformerDialPoint("dial-04", 192, 3, 6, 96, 12, 2048),
        TransformerDialPoint("dial-05", 256, 4, 8, 128, 12, 2048),
        TransformerDialPoint("dial-06", 320, 5, 8, 128, 8, 2048),
        TransformerDialPoint("dial-07", 384, 6, 8, 160, 8, 2048),
        TransformerDialPoint("dial-08", 512, 8, 8, 192, 6, 2048),
    )
    validate_dial(points)
    return points

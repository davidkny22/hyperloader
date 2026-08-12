"""Recognizable workload identities booked onto the loader-tax curve."""

from __future__ import annotations

from dataclasses import dataclass

REFERENCE_SYSTEMS = ("counterfactual", "torch", "hyperloader", "spdl")


@dataclass(frozen=True)
class NamedTrainingAnchor:
    """One real workload family and its cross-loader comparison controls."""

    anchor_id: str
    family: str
    model_name: str
    task: str
    data_class: str
    batch_size: int
    sequence_length: int | None
    input_resolution: int | None
    precision: str
    references: tuple[str, ...] = REFERENCE_SYSTEMS

    def validate(self) -> None:
        """Reject incomplete identities and inconsistent modality dimensions."""
        if self.family not in {"gpt-pretraining", "vision-finetuning"}:
            raise ValueError("named anchor family is unsupported")
        if not self.anchor_id or not self.model_name or self.batch_size <= 0:
            raise ValueError("named anchors require identity and a positive batch size")
        if self.references != REFERENCE_SYSTEMS:
            raise ValueError("named anchors require ceiling, Torch, hyperloader, and SPDL")
        if self.family == "gpt-pretraining":
            if self.sequence_length is None or self.input_resolution is not None:
                raise ValueError("GPT anchors require only a sequence length")
        elif self.input_resolution is None or self.sequence_length is not None:
            raise ValueError("vision anchors require only an input resolution")


def default_named_anchors() -> tuple[NamedTrainingAnchor, ...]:
    """Return two GPT-2 sizes and one ResNet-18 image-folder fine-tune."""
    anchors = (
        NamedTrainingAnchor(
            "gpt2-124m-pretraining",
            "gpt-pretraining",
            "GPT-2 124M",
            "next-token pretraining",
            "pretokenized-text",
            8,
            256,
            None,
            "bfloat16",
        ),
        NamedTrainingAnchor(
            "gpt2-355m-pretraining",
            "gpt-pretraining",
            "GPT-2 355M",
            "next-token pretraining",
            "pretokenized-text",
            2,
            256,
            None,
            "bfloat16",
        ),
        NamedTrainingAnchor(
            "resnet18-image-folder-finetuning",
            "vision-finetuning",
            "ResNet-18",
            "image-folder fine-tuning",
            "image-folder-standard-augmentation",
            64,
            None,
            224,
            "bfloat16",
        ),
    )
    for anchor in anchors:
        anchor.validate()
    return anchors

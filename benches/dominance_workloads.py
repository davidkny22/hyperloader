"""Six equal-input workloads for provisional loader comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_protocol.matrix import workload_names

SAMPLE_WIDTH = 512
DEFAULT_BATCHES = 32


def forbidden_decoder(_value: Any) -> Any:
    """Fail when the selected image provider does not replace the refuge."""
    raise AssertionError("the pinned image provider was not installed")


@dataclass(slots=True)
class DecodedImageDataset:
    """Decode one PNG tensor through the same pinned torchvision provider."""

    encoded: tuple[Any, ...]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, index: int) -> Any:
        from torchvision.io import decode_png

        return decode_png(self.encoded[index])


@dataclass(slots=True)
class WorkloadBundle:
    """System-specific surfaces over one immutable logical dataset."""

    name: str
    gpu_regime: str
    batch_size: int
    reference_dataset: Any
    hyperloader_dataset: Any
    collate_fn: Any
    stage_plan_pin: str
    mapped_owner: Any = None

    def normalize(self, value: Any) -> Any:
        """Return one dense tensor consumed identically by the GPU workload."""
        if self.name != "arrow-tabular":
            return value
        columns = value["tokens"]
        if isinstance(columns, list):
            import torch

            return torch.stack(columns, dim=1)
        return columns

    def close(self) -> None:
        """Release workload-owned mappings after all feeders close."""
        if self.mapped_owner is not None:
            self.mapped_owner._mmap.close()
            self.mapped_owner = None


def make_workload(
    name: str, root: Path, *, batches: int = DEFAULT_BATCHES
) -> WorkloadBundle:
    """Build one named matrix cell with identical source values for all systems."""
    if name not in workload_names():
        raise ValueError(f"unknown dominance workload {name!r}")
    if batches <= 0:
        raise ValueError("workload batch count must be positive")
    factories = {
        "images-light": _image_light,
        "images-heavy": _image_heavy,
        "fixed-text": _fixed_text,
        "varlen-text": _variable_text,
        "arrow-tabular": _arrow_tabular,
        "numpy-array": _numpy_array,
    }
    return factories[name](root, batches)


def _image_light(_root: Path, batches: int) -> WorkloadBundle:
    return _image_workload("images-light", edge=64, batch_size=64, batches=batches)


def _image_heavy(_root: Path, batches: int) -> WorkloadBundle:
    return _image_workload("images-heavy", edge=512, batch_size=8, batches=batches)


def _image_workload(
    name: str, *, edge: int, batch_size: int, batches: int
) -> WorkloadBundle:
    import torch
    from torchvision.io import encode_png

    from hyperloader import Collate, Decode, Source, pipeline

    seeds = []
    base = torch.arange(3 * edge * edge, dtype=torch.int64).reshape(3, edge, edge)
    for offset in range(batch_size):
        image = ((base * 37 + offset * 19) % 251).to(torch.uint8)
        seeds.append(encode_png(image))
    encoded = tuple(seeds * batches)
    reference = DecodedImageDataset(encoded)
    native = pipeline(
        Source(encoded, output_type=torch.Tensor),
        Decode(
            forbidden_decoder,
            input_type=torch.Tensor,
            output_type=torch.Tensor,
            codec="png",
            substitute=True,
        ),
        Collate(torch.stack, input_type=torch.Tensor, output_type=torch.Tensor),
    )
    return WorkloadBundle(
        name,
        "compute",
        batch_size,
        reference,
        native,
        torch.stack,
        "torchvision.io.decode_png@0.28.0; native final arena stack",
    )


def _fixed_text(_root: Path, batches: int) -> WorkloadBundle:
    import torch

    batch_size = 64
    rows = batch_size * batches
    source = torch.arange(SAMPLE_WIDTH, dtype=torch.int64).repeat(rows, 1)
    return WorkloadBundle(
        "fixed-text",
        "compute",
        batch_size,
        source,
        source,
        torch.stack,
        "contiguous pre-tokenized tensor view",
    )


def _variable_text(_root: Path, batches: int) -> WorkloadBundle:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    from hyperloader import Collate, Source, pipeline

    batch_size = 64
    values = tuple(
        torch.arange(256 + index % 257, dtype=torch.int64)
        for index in range(batch_size * batches)
    )
    native = pipeline(
        Source(values, output_type=torch.Tensor),
        Collate(pad_sequence, input_type=torch.Tensor, output_type=torch.Tensor),
    )
    return WorkloadBundle(
        "varlen-text",
        "compute",
        batch_size,
        values,
        native,
        pad_sequence,
        "pre-tokenized tensors; native variable padding into final arena slot",
    )


def _arrow_tabular(_root: Path, batches: int) -> WorkloadBundle:
    from datasets import Dataset
    from torch.utils.data import default_collate

    batch_size = 64
    rows = batch_size * batches
    values = np.arange(SAMPLE_WIDTH, dtype=np.int64)[None, :].repeat(rows, axis=0)
    dataset = Dataset.from_dict({"tokens": values.tolist()})
    return WorkloadBundle(
        "arrow-tabular",
        "bandwidth",
        batch_size,
        dataset,
        dataset,
        default_collate,
        "Hugging Face Arrow query; unformatted numeric batch conversion",
    )


def _numpy_array(root: Path, batches: int) -> WorkloadBundle:
    from torch.utils.data import default_collate

    batch_size = 64
    rows = batch_size * batches
    path = root / "numpy-array.bin"
    dataset = np.memmap(path, dtype=np.int64, mode="w+", shape=(rows, SAMPLE_WIDTH))
    dataset[:] = np.arange(SAMPLE_WIDTH, dtype=np.int64)
    dataset.flush()
    return WorkloadBundle(
        "numpy-array",
        "bandwidth",
        batch_size,
        dataset,
        dataset,
        default_collate,
        "C-order NumPy memmap storage view",
        mapped_owner=dataset,
    )

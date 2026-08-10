"""Per-family dataset fingerprint elements and strict content digests."""

from __future__ import annotations

import hashlib
import pickle
from typing import Any

from hyperloader.stages import Pipeline

from .arrays import array_elements
from .arrow import arrow_elements, parquet_elements
from .callable import callable_identity, stable_value
from .files import file_elements
from .model import ContractFingerprint, FingerprintElement


def build_dataset_fingerprint(dataset: Any, mode: str) -> ContractFingerprint:
    """Fingerprint one dataset in content or strict mode."""
    if mode not in {"content", "strict"}:
        raise ValueError("dataset fingerprint mode must be content or strict")
    elements = [FingerprintElement("dataset.fingerprint_mode", mode)]
    elements.extend(_dataset_elements(dataset, mode, "dataset"))
    return ContractFingerprint(tuple(elements))


def _dataset_elements(dataset: Any, mode: str, prefix: str) -> list[FingerprintElement]:
    kind = _qualified_type(dataset)
    elements = [FingerprintElement(f"{prefix}.type", kind)]
    if isinstance(dataset, Pipeline):
        elements.extend(_pipeline_elements(dataset, mode, prefix))
    elif _is_exact(dataset, "torch", "Tensor"):
        elements.extend(array_elements(dataset, mode, prefix))
    elif _is_exact(dataset, "torch.utils.data.dataset", "TensorDataset"):
        tensors = dataset.tensors
        elements.append(FingerprintElement(f"{prefix}.tensor_count", len(tensors)))
        for index, tensor in enumerate(tensors):
            elements.extend(array_elements(tensor, mode, f"{prefix}.tensors[{index}]"))
    elif _inherits(dataset, "torchvision.datasets.folder", "DatasetFolder"):
        elements.extend(_folder_elements(dataset, mode, prefix))
    elif _is_exact(dataset, "datasets.arrow_dataset", "Dataset"):
        elements.extend(arrow_elements(dataset, mode, prefix))
    elif _is_exact(dataset, "numpy", "memmap"):
        elements.extend(array_elements(dataset, mode, prefix))
        elements.append(
            FingerprintElement(f"{prefix}.offset", int(getattr(dataset, "offset", 0)))
        )
        elements.append(
            FingerprintElement(
                f"{prefix}.order",
                "F"
                if dataset.flags.f_contiguous and not dataset.flags.c_contiguous
                else "C",
            )
        )
        elements.extend(
            file_elements(
                [getattr(dataset, "filename", "")],
                mode,
                f"{prefix}.files",
            )
        )
    elif _is_exact(dataset, "pyarrow._dataset", "FileSystemDataset"):
        elements.extend(parquet_elements(dataset, mode, prefix))
    else:
        elements.extend(_generic_elements(dataset, mode, prefix))
    return elements


def _pipeline_elements(
    dataset: Pipeline[Any, Any], mode: str, prefix: str
) -> list[FingerprintElement]:
    elements = _dataset_elements(dataset.source.source, mode, f"{prefix}.source")
    elements.append(FingerprintElement(f"{prefix}.stage_count", len(dataset.stages)))
    for index, stage in enumerate(dataset.stages):
        stage_prefix = f"{prefix}.stages[{index}]"
        target = getattr(stage, "fn", getattr(stage, "source", None))
        elements.extend(
            [
                FingerprintElement(f"{stage_prefix}.type", _qualified_type(stage)),
                FingerprintElement(
                    f"{stage_prefix}.callable", callable_identity(target)
                ),
                FingerprintElement(
                    f"{stage_prefix}.input_type",
                    _qualified_type_token(getattr(stage, "input_type", object)),
                ),
                FingerprintElement(
                    f"{stage_prefix}.output_type",
                    _qualified_type_token(getattr(stage, "output_type", object)),
                ),
                FingerprintElement(
                    f"{stage_prefix}.io", stable_value(getattr(stage, "io", "none"))
                ),
                FingerprintElement(
                    f"{stage_prefix}.thread_safety",
                    stable_value(getattr(stage, "thread_safety", "isolated")),
                ),
            ]
        )
    return elements


def _folder_elements(dataset: Any, mode: str, prefix: str) -> list[FingerprintElement]:
    samples = tuple(getattr(dataset, "samples", ()))
    paths = [sample[0] for sample in samples]
    elements = file_elements(
        paths, mode, f"{prefix}.files", getattr(dataset, "root", None)
    )
    for index, sample in enumerate(samples):
        target = sample[1] if len(sample) > 1 else None
        elements.append(
            FingerprintElement(f"{prefix}.files[{index}].target", stable_value(target))
        )
    elements.extend(
        [
            FingerprintElement(
                f"{prefix}.loader", callable_identity(getattr(dataset, "loader", None))
            ),
            FingerprintElement(
                f"{prefix}.transform",
                callable_identity(getattr(dataset, "transform", None)),
            ),
            FingerprintElement(
                f"{prefix}.target_transform",
                callable_identity(getattr(dataset, "target_transform", None)),
            ),
        ]
    )
    return elements


def _generic_elements(dataset: Any, mode: str, prefix: str) -> list[FingerprintElement]:
    try:
        length: int | None = len(dataset)
    except TypeError:
        length = None
    elements = [
        FingerprintElement(f"{prefix}.length", length),
        FingerprintElement(
            f"{prefix}.getitem",
            callable_identity(getattr(type(dataset), "__getitem__", None)),
        ),
    ]
    if mode == "strict":
        try:
            payload = pickle.dumps(dataset, protocol=5)
        except (pickle.PickleError, TypeError, AttributeError):
            digest = "unavailable"
        else:
            digest = hashlib.sha256(payload).hexdigest()
        elements.append(FingerprintElement(f"{prefix}.content_sha256", digest))
    return elements


def _qualified_type(value: Any) -> str:
    return _qualified_type_token(type(value))


def _qualified_type_token(value: type[Any]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _is_exact(value: Any, module: str, name: str) -> bool:
    value_type = type(value)
    return value_type.__module__ == module and value_type.__name__ == name


def _inherits(value: Any, module: str, name: str) -> bool:
    return any(
        base.__module__ == module and base.__name__ == name
        for base in type(value).__mro__
    )

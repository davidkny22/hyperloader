"""Result-contract fingerprint assembly from one validated loader plan."""

from __future__ import annotations

from typing import Any

from hyperloader.config import AUTO
from hyperloader.planner import TensorPlan
from hyperloader.stages import Pipeline

from .callable import callable_identity, stable_value
from .model import ContractFingerprint, FingerprintElement

CONTRACT_VERSION = 1


def build_contract_fingerprint(loader: Any) -> ContractFingerprint:
    """Collect every result-observable input in stable diagnostic order."""
    dataset_fingerprint = loader._dataset_fingerprint
    elements = [FingerprintElement("contract_version", CONTRACT_VERSION)]
    elements.extend(dataset_fingerprint.elements)
    map_style = loader._plan is not None
    elements.append(FingerprintElement("plan.kind", "map" if map_style else "iterable"))

    sampler = (
        loader.batch_sampler if loader.batch_sampler is not None else loader.sampler
    )
    if sampler is None:
        elements.append(FingerprintElement("sampler.kind", "native"))
        elements.append(FingerprintElement("sampler.shuffle", bool(loader.shuffle)))
        global_batch = (loader.batch_size or 1) * _world_size(loader)
    else:
        kind = "batch-sampler" if loader.batch_sampler is not None else "user"
        elements.append(FingerprintElement("sampler.kind", kind))
        elements.append(
            FingerprintElement("sampler.identity", callable_identity(sampler))
        )
        global_batch = 0
    elements.append(FingerprintElement("placement.B_g", global_batch))

    if map_style:
        elements.append(
            FingerprintElement(
                "placement.exact_count", loader.config.determinism.exact_count
            )
        )
    else:
        lanes = loader.num_workers if isinstance(loader.num_workers, int) else 1
        elements.extend(
            [
                FingerprintElement("iterable.L", max(1, lanes)),
                FingerprintElement("iterable.W", _world_size(loader)),
                FingerprintElement("iterable.batch_size", loader.batch_size),
                FingerprintElement(
                    "iterable.worker_init_fn",
                    callable_identity(loader.worker_init_fn),
                ),
            ]
        )

    elements.append(FingerprintElement("decoder_pins", _decoder_pins(loader)))
    seeded_libs = loader.config.determinism.seeded_libs
    if seeded_libs is not AUTO:
        elements.append(FingerprintElement("seeded_libs", stable_value(seeded_libs)))
    batch_shape = _batch_shape(loader)
    placement = getattr(loader, "_map_placement", None)
    if map_style and sampler is None and placement is not None and placement.enabled:
        batch_shape = _elastic_batch_shape(batch_shape, loader.batch_size or 1)
    elements.extend(
        [
            FingerprintElement("mode", loader.mode),
            FingerprintElement("delivery", loader.delivery),
            FingerprintElement("drop_last", loader.drop_last),
            FingerprintElement("collate.identity", _collate_identity(loader)),
            FingerprintElement("batch_shape", batch_shape),
        ]
    )
    return ContractFingerprint(tuple(elements))


def _world_size(loader: Any) -> int:
    configured = loader.config.distributed.world_size
    return configured if isinstance(configured, int) else 1


def _decoder_pins(loader: Any) -> Any:
    return [selection.to_dict() for selection in loader._decoder_selections]


def _collate_identity(loader: Any) -> Any:
    if loader.collate_fn is not None:
        return callable_identity(loader.collate_fn)
    if isinstance(loader.dataset, Pipeline):
        return callable_identity(loader.dataset.collate_stage.fn)
    return {"anchor": "engine-default", "contract_version": CONTRACT_VERSION}


def _batch_shape(loader: Any) -> Any:
    declared = loader.config.memory.batch_shape
    if declared is not AUTO:
        return {"source": "declared", "value": stable_value(declared)}
    native = getattr(loader, "_native_batch_shape", None)
    if native is not None:
        return native
    pool = getattr(loader, "_process_pool", None)
    if pool is not None:
        return pool.batch_shape_fingerprint
    if isinstance(loader._plan, TensorPlan) and loader._plan.length:
        sample = loader.dataset[0]
        shape = [int(value) for value in sample.shape]
        if loader.batch_size is not None:
            shape.insert(0, loader.batch_size)
        return {"dtype": str(sample.dtype), "shape": shape, "source": "derived"}
    return {"source": "probe-pending"}


def _elastic_batch_shape(value: Any, batch_size: int) -> Any:
    """Replace topology-derived batch extents with a stable placement marker."""
    if isinstance(value, list):
        return [_elastic_batch_shape(item, batch_size) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {
        key: _elastic_batch_shape(item, batch_size) for key, item in value.items()
    }
    shape = normalized.get("shape")
    if isinstance(shape, list) and shape and shape[0] == batch_size:
        normalized["shape"] = [{"placement": "per-rank-batch"}, *shape[1:]]
    if normalized.get("length") == batch_size:
        normalized["length"] = {"placement": "per-rank-batch"}
    if normalized.get("batch_size") == batch_size:
        normalized["batch_size"] = {"placement": "per-rank-batch"}
    return normalized

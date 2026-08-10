"""Result-contract fingerprint assembly from one validated loader plan."""

from __future__ import annotations

import sys
from typing import Any

from hyperloader.config import AUTO
from hyperloader.planner import TensorPlan
from hyperloader.stages import Decode, Pipeline

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
    elements.extend(
        [
            FingerprintElement("mode", loader.mode),
            FingerprintElement("delivery", loader.delivery),
            FingerprintElement("drop_last", loader.drop_last),
            FingerprintElement("collate.identity", _collate_identity(loader)),
            FingerprintElement("batch_shape", _batch_shape(loader)),
        ]
    )
    return ContractFingerprint(tuple(elements))


def _world_size(loader: Any) -> int:
    configured = loader.config.distributed.world_size
    return configured if isinstance(configured, int) else 1


def _decoder_pins(loader: Any) -> Any:
    configured = loader.config.determinism.decoder_pins
    if configured is not AUTO:
        return stable_value(configured)
    pins = []
    if isinstance(loader.dataset, Pipeline):
        for index, stage in enumerate(loader.dataset.sample_stages):
            if isinstance(stage, Decode):
                pins.append(
                    {
                        "name": f"pipeline-decode-{index}",
                        "version": callable_identity(stage.fn),
                    }
                )
    return {"pins": pins, "platform": sys.platform, "source": "platform-default"}


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

"""Constructor validation and option precedence for the public loader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AUTO, Auto, HyperConfig
from .stages import Pipeline


@dataclass(frozen=True, slots=True)
class ResolvedOptions:
    """Carry validated values that differ from direct constructor arguments."""

    config: HyperConfig
    seed: int | None
    delivery: str
    delivery_memory: str


def validate_constructor(
    dataset: Any,
    batch_size: int | None,
    shuffle: bool | None,
    sampler: Any,
    batch_sampler: Any,
    num_workers: int | Auto,
    collate_fn: Any,
    pin_memory: bool,
    drop_last: bool,
    timeout: float,
    prefetch_factor: int | None | Auto,
    thread_safe: bool,
    mode: str,
    in_order: bool,
    delivery: str | Auto,
    seed: int | None,
    device: str | None,
    config: HyperConfig | None,
) -> ResolvedOptions:
    """Validate constructor conflicts and resolve public precedence rules."""
    if sampler is not None and shuffle:
        raise ValueError("sampler option is mutually exclusive with shuffle")
    if batch_sampler is not None and (
        batch_size != 1 or shuffle or sampler is not None or drop_last
    ):
        raise ValueError(
            "batch_sampler is mutually exclusive with batch_size, shuffle, sampler, "
            "and drop_last"
        )
    if batch_size is not None and (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer or None")
    if batch_size is None and drop_last:
        raise ValueError("batch_size=None is mutually exclusive with drop_last")
    if isinstance(dataset, Pipeline) and collate_fn is not None:
        raise ValueError("pipeline Collate is mutually exclusive with collate_fn")

    _require_nonnegative_workers(num_workers)
    if timeout < 0:
        raise ValueError("timeout must be nonnegative")
    if prefetch_factor is not AUTO and prefetch_factor is not None:
        if isinstance(prefetch_factor, bool) or prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be auto, None, or positive")
        if num_workers == 0:
            raise ValueError(
                "prefetch_factor can only be specified when num_workers is positive"
            )
    if mode not in {"native", "torch-compat"}:
        raise ValueError("mode must be native or torch-compat")
    if not isinstance(thread_safe, bool):
        raise TypeError("thread_safe must be a boolean declaration")

    resolved_config = config if config is not None else HyperConfig()
    if (
        seed is not None
        and resolved_config.seed is not None
        and seed != resolved_config.seed
    ):
        raise ValueError("seed conflicts with config.seed")
    resolved_seed = seed if seed is not None else resolved_config.seed

    process_ceiling = resolved_config.executor.process_ceiling
    if (
        process_ceiling is not AUTO
        and num_workers is not AUTO
        and process_ceiling != num_workers
    ):
        raise ValueError("num_workers conflicts with config.executor.process_ceiling")
    if mode == "torch-compat" and resolved_config.executor.on_worker_death == "restart":
        raise ValueError("worker restart is unavailable in torch-compat mode")

    resolved_delivery = _resolve_delivery(in_order, delivery)
    configured_memory = resolved_config.memory.delivery_memory
    requested_memory = (
        "device" if device is not None else "pinned" if pin_memory else None
    )
    if (
        configured_memory != "auto"
        and requested_memory is not None
        and configured_memory != requested_memory
    ):
        raise ValueError("memory.delivery_memory conflicts with pin_memory or device")
    resolved_memory = (
        configured_memory
        if configured_memory != "auto"
        else requested_memory
        if requested_memory is not None
        else "auto"
    )
    return ResolvedOptions(
        config=resolved_config,
        seed=resolved_seed,
        delivery=resolved_delivery,
        delivery_memory=resolved_memory,
    )


def _require_nonnegative_workers(num_workers: int | Auto) -> None:
    if num_workers is AUTO:
        return
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise TypeError("num_workers must be auto or an integer")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")


def _resolve_delivery(in_order: bool, delivery: str | Auto) -> str:
    expected = "in-order" if in_order else "on-completion"
    if delivery is AUTO or delivery == "auto":
        return expected
    if delivery not in {"in-order", "on-completion"}:
        raise ValueError("delivery must be auto, in-order, or on-completion")
    if delivery != expected:
        raise ValueError(f"delivery={delivery!r} conflicts with in_order={in_order!r}")
    return delivery

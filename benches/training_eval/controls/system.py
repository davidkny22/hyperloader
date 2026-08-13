"""Assembly of system-state evidence for paired training cells."""

from __future__ import annotations

import gc
import os
from typing import Any

from ..models import TrainingCellConfig, TrainingEnvironment
from .host import (
    affinity,
    background_load,
    cpu_clocks,
    cpu_governors,
    idle_states,
    interrupt_homes,
    memory_pressure,
    nvidia_query,
    storage_state,
    thermal_zones,
    thread_count,
)
from .runtime import allocator_state, library_versions, torch_threading
from .worker_probe import THREAD_ENVIRONMENT_KEYS


def capture_system_state(
    config: TrainingCellConfig, environment: TrainingEnvironment
) -> dict[str, Any]:
    """Capture every non-feeder controlled-variable family."""
    return {
        "allocator_state": allocator_state(),
        "background_load": background_load(),
        "batch_and_delivery": {
            "batch_size": config.batch_size,
            "delivery": config.delivery,
            "device": config.device,
            "input_resolution": config.input_resolution,
            "sequence_length": config.sequence_length,
        },
        "cache_state": {
            "gc_counts": list(gc.get_count()),
            "resident_batches": config.resident_batches,
            "warmup_steps": config.warmup_steps,
        },
        "cpu_clocks": cpu_clocks(),
        "dataset_and_rng": {
            "dataset_identity": config.dataset_identity,
            "dataset_rows": config.dataset_rows,
            "seed": config.seed,
        },
        "feeder_tuning": {
            "reference": {
                "prefetch": config.reference_prefetch,
                "workers": config.reference_workers,
            },
            "subject": {
                "prefetch": config.subject_prefetch,
                "workers": config.subject_workers,
            },
            "tuning_knobs": list(config.tuning_knobs),
            "tuning_seconds": config.tuning_seconds,
            "tuning_trials": config.tuning_trials,
        },
        "governor_and_power": {
            "declared_cpu_governor": environment.cpu_governor,
            "observed_cpu_governors": cpu_governors(),
            "power_profile": environment.power_profile,
            "plugged_in": environment.plugged_in,
        },
        "gpu_clocks": {
            "declared_accelerator_clock": environment.accelerator_clock,
            "declared_memory_clock": environment.memory_clock,
            "observed": nvidia_query(
                (
                    "clocks.current.graphics",
                    "clocks.current.memory",
                    "pstate",
                    "power.draw",
                )
            ),
        },
        "idle_states": idle_states(),
        "interrupt_homes": interrupt_homes(),
        "lease_and_interactivity": {
            "interactive_load": environment.interactive_load,
            "lease_kind": environment.lease_kind,
            "lease_token": environment.lease_token,
        },
        "library_versions": library_versions(),
        "machine_keeper": {
            "active_microseconds": environment.machine_state_active_microseconds,
            "control": environment.machine_state_control,
            "cpus": list(environment.machine_state_cpus),
            "period_microseconds": environment.machine_state_period_microseconds,
        },
        "memory_pressure": memory_pressure(),
        "pair_order_and_duration": {
            "alternating_order": True,
            "half_seconds": config.half_seconds,
            "reference": config.reference,
            "subject": config.subject,
        },
        "process_affinity": {"consumer": affinity(0)},
        "process_thread_counts": {
            "consumer": thread_count("/proc/self/status")
        },
        "storage_state": storage_state(),
        "thermal_state": {
            "accelerator_temperature_celsius": nvidia_query(("temperature.gpu",)),
            "declared_steady": environment.thermal_steady,
            "platform_temperatures_millicelsius": thermal_zones(),
        },
        "thread_environment": {
            key: os.environ.get(key) for key in THREAD_ENVIRONMENT_KEYS
        },
        "torch_threading": torch_threading(),
        "workload_definition": {
            "attention_heads": config.attention_heads,
            "learning_rate": config.learning_rate,
            "model_depth": config.model_depth,
            "model_name": config.model_name,
            "model_parameters": config.model_parameters,
            "model_width": config.model_width,
            "optimizer": config.optimizer,
            "precision": config.precision,
            "workload_family": config.workload_family,
        },
    }

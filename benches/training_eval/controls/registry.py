"""Stable classification of paired-cell controlled variables."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlledVariableDefinition:
    """Name one variable family and its experimental treatment."""

    name: str
    classification: str
    rationale: str


CONTROLLED_VARIABLE_REGISTRY = (
    ControlledVariableDefinition(
        "workload_definition",
        "pinned",
        "Model structure, parameter count, precision, optimizer, and learning rate are fixed for a point.",
    ),
    ControlledVariableDefinition(
        "dataset_and_rng",
        "pinned",
        "Source identity, finite row count, and root seed are fixed for both feeders.",
    ),
    ControlledVariableDefinition(
        "batch_and_delivery",
        "matched",
        "Batch shape, delivery memory, device, and transfer mode are equal across halves.",
    ),
    ControlledVariableDefinition(
        "feeder_tuning",
        "deliberately-varying",
        "Each loader receives its own equal-budget worker and prefetch tuning result.",
    ),
    ControlledVariableDefinition(
        "pair_order_and_duration",
        "matched",
        "The live process alternates feeder order and applies one half-duration rule.",
    ),
    ControlledVariableDefinition(
        "process_thread_counts",
        "logged",
        "Consumer and feeder process thread counts expose oversubscription and passive pools.",
    ),
    ControlledVariableDefinition(
        "torch_threading",
        "pinned",
        "Process workers must start dataset code with one Torch intra-op thread.",
    ),
    ControlledVariableDefinition(
        "thread_environment",
        "matched",
        "OMP, MKL, OpenBLAS, NumExpr, and Torch threading inputs are inherited and recorded per process.",
    ),
    ControlledVariableDefinition(
        "process_affinity",
        "logged",
        "Consumer and worker CPU masks reveal placement and unintended confinement.",
    ),
    ControlledVariableDefinition(
        "machine_keeper",
        "matched",
        "The same native keeper cores, active interval, and period surround both halves.",
    ),
    ControlledVariableDefinition(
        "cpu_clocks",
        "logged",
        "Observed CPU frequency state is sampled without assuming one topology or frequency.",
    ),
    ControlledVariableDefinition(
        "gpu_clocks",
        "pinned",
        "The external clock guard applies one resolved accelerator clock to the complete cell.",
    ),
    ControlledVariableDefinition(
        "governor_and_power",
        "pinned",
        "The declared CPU governor and power profile must match the active machine state.",
    ),
    ControlledVariableDefinition(
        "idle_states",
        "logged",
        "Available idle states, disable flags, residency, and usage expose wake-state drift.",
    ),
    ControlledVariableDefinition(
        "interrupt_homes",
        "logged",
        "Accelerator interrupt counters identify the cores participating in wake delivery.",
    ),
    ControlledVariableDefinition(
        "library_versions",
        "pinned",
        "The installed product and training-library identities bind the executable surface.",
    ),
    ControlledVariableDefinition(
        "allocator_state",
        "logged",
        "CUDA allocator configuration and live allocation totals expose memory-state drift.",
    ),
    ControlledVariableDefinition(
        "cache_state",
        "matched",
        "Both feeders receive the same warmup rule while host and device cache indicators are recorded.",
    ),
    ControlledVariableDefinition(
        "thermal_state",
        "logged",
        "Thermal readiness is asserted and available device and platform temperatures are recorded.",
    ),
    ControlledVariableDefinition(
        "background_load",
        "logged",
        "Load averages, process counts, and accelerator clients expose concurrent activity.",
    ),
    ControlledVariableDefinition(
        "lease_and_interactivity",
        "pinned",
        "Exclusive lease ownership and the no-interactive-load assertion bind the measurement window.",
    ),
    ControlledVariableDefinition(
        "memory_pressure",
        "logged",
        "Host memory and file-cache totals expose pressure that can change loader throughput.",
    ),
    ControlledVariableDefinition(
        "storage_state",
        "logged",
        "Filesystem and source-path metadata expose storage placement without pinning one machine path.",
    ),
)


def validate_registry() -> None:
    """Reject missing, duplicate, or invalid classifications."""
    names = [entry.name for entry in CONTROLLED_VARIABLE_REGISTRY]
    if len(names) != len(set(names)) or not names:
        raise ValueError("controlled-variable registry names must be unique")
    allowed = {"pinned", "matched", "logged", "deliberately-varying"}
    if any(entry.classification not in allowed for entry in CONTROLLED_VARIABLE_REGISTRY):
        raise ValueError("controlled-variable classification is invalid")

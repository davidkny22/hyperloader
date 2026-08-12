"""Pure-Python refuge for the optional native extension module."""

from .fallback import native as _native

IS_FALLBACK = _native.IS_FALLBACK
_CostProfile = _native.CostProfile
_MachineKeeper = _native.MachineKeeper
_ProcessResources = _native.ProcessResources
_StaticSchedule = _native.StaticSchedule
_Telemetry = _native.Telemetry
_WorkerCommand = _native.WorkerCommand
_WorkerEndpoint = _native.WorkerEndpoint
_current_cpu = _native.current_cpu
_default_collate = _native.default_collate
_feistel_permute = _native.feistel_permute
_materialized_permutation = _native.materialized_permutation
_permutation_index = _native.permutation_index
_rank_placements = _native.rank_placements
_rng_block = _native.rng_block
_rng_block_from_key = _native.rng_block_from_key
_sample_rng_context = _native.sample_rng_context
package_version = _native.package_version

__all__ = ["IS_FALLBACK", "package_version"]

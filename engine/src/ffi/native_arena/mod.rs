//! Native-thread final-slot allocation over the shared arena substrate.

mod buffer;

use crate::arena::{ArenaAllocator, GrowthPolicy, RegionRegistry, RegionToken, SlabSpec};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::path::PathBuf;

use buffer::NativeSlot;

#[pyclass(name = "_NativeArena")]
pub(super) struct NativeArena {
    allocator: ArenaAllocator,
}

#[pymethods]
impl NativeArena {
    #[new]
    #[pyo3(signature = (initial_capacity, slot_count, growth, registry_path=None))]
    fn new(
        initial_capacity: usize,
        slot_count: u32,
        growth: &str,
        registry_path: Option<PathBuf>,
    ) -> PyResult<Self> {
        if initial_capacity == 0 || slot_count == 0 {
            return Err(PyValueError::new_err(
                "native arena capacity and slot count must be positive",
            ));
        }
        let overflow_capacity = initial_capacity
            .checked_mul(2)
            .ok_or_else(|| PyValueError::new_err("native arena capacity overflowed"))?;
        let policy = match growth {
            "safe" => GrowthPolicy::Safe,
            "strict-error" => GrowthPolicy::StrictError,
            _ => return Err(PyValueError::new_err("unknown native arena growth policy")),
        };
        let registry = match registry_path {
            Some(path) => RegionRegistry::new(path),
            None => {
                RegionRegistry::prepare_current_user()
                    .map_err(runtime_error)?
                    .0
            }
        };
        let allocator = ArenaAllocator::new(
            registry,
            RegionToken::random().map_err(runtime_error)?,
            &[
                SlabSpec {
                    slot_capacity: initial_capacity,
                    slots_per_slab: slot_count,
                    overflow: false,
                },
                SlabSpec {
                    slot_capacity: overflow_capacity,
                    slots_per_slab: 1,
                    overflow: true,
                },
            ],
            policy,
        )
        .map_err(runtime_error)?;
        Ok(Self { allocator })
    }

    fn reserve(&self, required: usize) -> PyResult<NativeSlot> {
        NativeSlot::reserve(self.allocator.clone(), required)
    }

    fn stats(&self) -> PyResult<(usize, u64, u64, u64)> {
        let stats = self.allocator.stats().map_err(runtime_error)?;
        Ok((
            stats.regions,
            stats.growth_events,
            stats.hold_events,
            stats.overflow_events,
        ))
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeArena>()?;
    module.add_class::<NativeSlot>()?;
    Ok(())
}

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

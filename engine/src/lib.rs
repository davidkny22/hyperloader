//! Native execution engine for hyperloader.

pub mod arena;
pub mod collate;
pub mod control;
pub mod error;
pub mod exec;
mod ffi;
pub mod io;
pub mod rng;
pub mod sched;
pub mod state;
pub mod telemetry;

use pyo3::prelude::*;

/// Initialize the native hyperloader extension.
#[pymodule(gil_used = false)]
fn _hyperloader(module: &Bound<'_, PyModule>) -> PyResult<()> {
    ffi::register(module)
}

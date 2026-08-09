//! Native execution engine for hyperloader.

pub mod arena;
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
#[pymodule]
fn _hyperloader(module: &Bound<'_, PyModule>) -> PyResult<()> {
    ffi::register(module)
}

#[cfg(test)]
mod tests {
    use super::ffi::package_version;

    #[test]
    fn package_version_matches_manifest() {
        assert_eq!(package_version(), env!("CARGO_PKG_VERSION"));
    }
}

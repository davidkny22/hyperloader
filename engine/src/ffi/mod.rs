//! Python extension registration is centralized in this module.

use pyo3::prelude::*;

/// Return the package version embedded in the native extension.
#[pyfunction]
pub(crate) fn package_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Register the stable native functions exposed by the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(package_version, module)?)?;
    Ok(())
}

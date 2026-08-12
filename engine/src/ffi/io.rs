//! Installed extension seams for plan-time I/O selection and range reads.

use crate::io::{BackendPreference, PlatformBackend};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use std::path::PathBuf;
use std::str::FromStr;

#[pyfunction(name = "_io_backend_kind")]
fn io_backend_kind(preference: &str) -> PyResult<&'static str> {
    let preference = BackendPreference::from_str(preference).map_err(value_error)?;
    PlatformBackend::select(preference)
        .map(|backend| backend.kind().as_str())
        .map_err(value_error)
}

#[pyfunction(name = "_read_range")]
#[pyo3(signature = (path, offset, length, backend="auto"))]
fn read_range(
    py: Python<'_>,
    path: PathBuf,
    offset: u64,
    length: usize,
    backend: &str,
) -> PyResult<Vec<u8>> {
    let preference = BackendPreference::from_str(backend).map_err(value_error)?;
    let reader = PlatformBackend::select(preference).map_err(value_error)?;
    py.detach(|| reader.read_range(&path, offset, length))
        .map_err(|error| PyOSError::new_err(error.to_string()))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(io_backend_kind, module)?)?;
    module.add_function(wrap_pyfunction!(read_range, module)?)?;
    Ok(())
}

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

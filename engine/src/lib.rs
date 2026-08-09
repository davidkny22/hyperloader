use pyo3::prelude::*;

/// Return the package version embedded in the native extension.
#[pyfunction]
fn package_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Initialize the native hyperloader extension.
#[pymodule]
fn _hyperloader(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(package_version, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::package_version;

    #[test]
    fn package_version_matches_manifest() {
        assert_eq!(package_version(), env!("CARGO_PKG_VERSION"));
    }
}

//! Private Python binding for bounded execution-cost profiles.

use crate::state::profile::{CostProfile, ProfileError};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::PathBuf;

/// Native profile storage and statistics exposed to the Python shell.
#[pyclass(name = "_CostProfile")]
pub(crate) struct PyCostProfile {
    profile: CostProfile,
}

#[pymethods]
impl PyCostProfile {
    #[new]
    fn new(position_count: u64, max_bytes: u64, alpha: f64) -> PyResult<Self> {
        Ok(Self {
            profile: CostProfile::new(position_count, max_bytes, alpha).map_err(value_error)?,
        })
    }

    #[staticmethod]
    fn load(path: PathBuf, position_count: u64, max_bytes: u64, alpha: f64) -> PyResult<Self> {
        Ok(Self {
            profile: CostProfile::load(&path, position_count, max_bytes, alpha)
                .map_err(value_error)?,
        })
    }

    fn observe(&mut self, position: u64, cost_ns: u64) -> PyResult<()> {
        self.profile.observe(position, cost_ns).map_err(value_error)
    }

    fn estimate(&self, position: u64) -> PyResult<Option<f64>> {
        self.profile.estimate(position).map_err(value_error)
    }

    fn statistics(&self) -> Option<(f64, f64, usize)> {
        self.profile
            .statistics()
            .map(|value| (value.mean_ns, value.p999_ns, value.populated))
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.profile.save(&path).map_err(value_error)
    }

    #[getter]
    fn degraded(&self) -> bool {
        self.profile.is_degraded()
    }

    #[getter]
    fn payload_bytes(&self) -> u64 {
        self.profile.payload_bytes()
    }
}

fn value_error(error: ProfileError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

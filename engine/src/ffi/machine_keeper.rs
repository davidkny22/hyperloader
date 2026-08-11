//! Python ownership seam for the native consumer machine keeper.

use crate::control::MachineKeeper;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Mutex;

#[pyclass(name = "_MachineKeeper")]
pub(crate) struct PyMachineKeeper {
    keeper: Mutex<Option<MachineKeeper>>,
}

#[pymethods]
impl PyMachineKeeper {
    #[new]
    fn new(
        cpus: Vec<usize>,
        maximum_duty: f64,
        initial_duty: f64,
        minimum_gap_ns: u64,
    ) -> PyResult<Self> {
        let keeper = MachineKeeper::new(cpus, maximum_duty, initial_duty, minimum_gap_ns)
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self {
            keeper: Mutex::new(Some(keeper)),
        })
    }

    fn observe_gap(&self, nanoseconds: u64) {
        if let Some(keeper) = self
            .keeper
            .lock()
            .expect("machine-keeping mutex poisoned")
            .as_ref()
        {
            keeper.observe_gap(nanoseconds);
        }
    }

    fn park(&self) {
        if let Some(keeper) = self
            .keeper
            .lock()
            .expect("machine-keeping mutex poisoned")
            .as_ref()
        {
            keeper.park();
        }
    }

    fn defer_park(&self, nanoseconds: u64) {
        if let Some(keeper) = self
            .keeper
            .lock()
            .expect("machine-keeping mutex poisoned")
            .as_ref()
        {
            keeper.defer_park(nanoseconds);
        }
    }

    fn duty(&self) -> f64 {
        self.keeper
            .lock()
            .expect("machine-keeping mutex poisoned")
            .as_ref()
            .map_or(0.0, MachineKeeper::duty)
    }

    fn close(&self) {
        if let Some(mut keeper) = self
            .keeper
            .lock()
            .expect("machine-keeping mutex poisoned")
            .take()
        {
            keeper.close();
        }
    }
}

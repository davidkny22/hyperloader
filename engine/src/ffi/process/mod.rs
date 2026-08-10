//! Private Python bindings for persistent process-owner and worker endpoints.

mod delivery;
mod owner;
mod sizing;
mod worker;

use pyo3::prelude::*;

pub(crate) use owner::ProcessResources;
pub(crate) use worker::{WorkerCommand, WorkerEndpoint};

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ProcessResources>()?;
    module.add_class::<WorkerEndpoint>()?;
    module.add_class::<WorkerCommand>()?;
    Ok(())
}

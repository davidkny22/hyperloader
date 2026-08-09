//! Recursive mapping and sequence collation.

use super::dispatch::default_collate;
use super::kind::is_instance;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub(super) fn collated_columns<'py>(
    py: Python<'py>,
    batch: &Bound<'py, PyAny>,
    width: usize,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let mut columns = Vec::with_capacity(width);
    for index in 0..width {
        let samples = PyList::empty(py);
        for item in batch.try_iter()? {
            samples.append(item?.get_item(index)?)?;
        }
        columns.push(default_collate(py, samples.as_any())?);
    }
    Ok(columns)
}

pub(super) fn mapping_collate<'py>(
    py: Python<'py>,
    batch: &Bound<'py, PyAny>,
    element: &Bound<'py, PyAny>,
    mutable: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let values = PyDict::new(py);
    for key in element.try_iter()? {
        let key = key?;
        let samples = PyList::empty(py);
        for item in batch.try_iter()? {
            samples.append(item?.get_item(&key)?)?;
        }
        values.set_item(&key, default_collate(py, samples.as_any())?)?;
    }

    let attempt = if mutable {
        let clone = PyModule::import(py, "copy")?
            .getattr("copy")?
            .call1((element,))?;
        clone.call_method1("update", (&values,))?;
        Ok(clone)
    } else {
        element.get_type().call1((&values,))
    };
    match attempt {
        Ok(output) => Ok(output),
        Err(error) if error.is_instance_of::<PyTypeError>(py) => Ok(values.into_any()),
        Err(error) => Err(error),
    }
}

pub(super) fn sequence_collate<'py>(
    py: Python<'py>,
    batch: &Bound<'py, PyAny>,
    element: &Bound<'py, PyAny>,
    builtins: &Bound<'py, PyModule>,
    mutable_sequence: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let width = element.len()?;
    for item in batch.try_iter()?.skip(1) {
        if item?.len()? != width {
            return Err(PyRuntimeError::new_err(
                "each element in list of batch should be of equal size",
            ));
        }
    }
    let columns = collated_columns(py, batch, width)?;
    if is_instance(builtins, element, &builtins.getattr("tuple")?)? {
        let output = PyList::empty(py);
        for value in columns {
            output.append(value)?;
        }
        return Ok(output.into_any());
    }

    let attempt = if is_instance(builtins, element, mutable_sequence)? {
        let clone = PyModule::import(py, "copy")?
            .getattr("copy")?
            .call1((element,))?;
        for (index, value) in columns.iter().enumerate() {
            clone.set_item(index, value)?;
        }
        Ok(clone)
    } else {
        let values = PyList::empty(py);
        for value in &columns {
            values.append(value)?;
        }
        element.get_type().call1((values,))
    };
    match attempt {
        Ok(output) => Ok(output),
        Err(error) if error.is_instance_of::<PyTypeError>(py) => {
            let output = PyList::empty(py);
            for value in columns {
                output.append(value)?;
            }
            Ok(output.into_any())
        }
        Err(error) => Err(error),
    }
}

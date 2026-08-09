//! Default collation mirrors torch's pinned recursive type contract.

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};

fn is_instance(
    builtins: &Bound<'_, PyModule>,
    value: &Bound<'_, PyAny>,
    class: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    builtins
        .getattr("isinstance")?
        .call1((value, class))?
        .extract()
}

fn collated_columns<'py>(
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

fn mapping_collate<'py>(
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

fn sequence_collate<'py>(
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

fn tensor_collate<'py>(
    batch: &Bound<'py, PyAny>,
    element: &Bound<'py, PyAny>,
    torch: &Bound<'py, PyModule>,
) -> PyResult<Bound<'py, PyAny>> {
    if element.getattr("is_nested")?.extract::<bool>()? {
        return Err(PyRuntimeError::new_err(
            "Batches of nested tensors are not currently supported by the default collate_fn; \
             please provide a custom collate_fn to handle them appropriately.",
        ));
    }
    let layout = element.getattr("layout")?;
    for name in [
        "sparse_coo",
        "sparse_csr",
        "sparse_bsr",
        "sparse_csc",
        "sparse_bsc",
    ] {
        if layout.eq(torch.getattr(name)?)? {
            return Err(PyRuntimeError::new_err(
                "Batches of sparse tensors are not currently supported by the default collate_fn; \
                 please provide a custom collate_fn to handle them appropriately.",
            ));
        }
    }
    torch.getattr("stack")?.call1((batch, 0))
}

fn custom_registration_matches(
    py: Python<'_>,
    element: &Bound<'_, PyAny>,
    builtins: &Bound<'_, PyModule>,
) -> PyResult<bool> {
    let module = PyModule::import(py, "torch.utils.data._utils.collate")?;
    let map = module.getattr("default_collate_fn_map")?;
    for key in map.call_method0("keys")?.try_iter()? {
        if is_instance(builtins, element, &key?)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Collate one batch according to torch's default recursive contract.
pub fn default_collate<'py>(
    py: Python<'py>,
    batch: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let element = batch.get_item(0)?;
    let torch = PyModule::import(py, "torch")?;
    let builtins = PyModule::import(py, "builtins")?;
    let collections = PyModule::import(py, "collections.abc")?;

    if is_instance(&builtins, &element, &torch.getattr("Tensor")?)? {
        return tensor_collate(batch, &element, &torch);
    }

    if let Ok(numpy) = PyModule::import(py, "numpy") {
        if is_instance(&builtins, &element, &numpy.getattr("ndarray")?)? {
            let dtype: String = element.getattr("dtype")?.getattr("str")?.extract()?;
            if dtype
                .chars()
                .any(|character| matches!(character, 'S' | 'a' | 'U' | 'O'))
            {
                return Err(PyTypeError::new_err(format!(
                    "default_collate: batch must contain tensors, numpy arrays, numbers, \
                     dicts or lists; found {}",
                    element.getattr("dtype")?.str()?
                )));
            }
            let converted = PyList::empty(py);
            for item in batch.try_iter()? {
                converted.append(torch.getattr("as_tensor")?.call1((item?,))?)?;
            }
            return default_collate(py, converted.as_any());
        }
        let scalar_types = PyTuple::new(
            py,
            [
                numpy.getattr("bool_")?,
                numpy.getattr("number")?,
                numpy.getattr("object_")?,
            ],
        )?;
        if is_instance(&builtins, &element, scalar_types.as_any())? {
            return torch.getattr("as_tensor")?.call1((batch,));
        }
    }

    if is_instance(&builtins, &element, &builtins.getattr("float")?)? {
        let keywords = PyDict::new(py);
        keywords.set_item("dtype", torch.getattr("float64")?)?;
        return torch.getattr("tensor")?.call((batch,), Some(&keywords));
    }
    if is_instance(&builtins, &element, &builtins.getattr("int")?)? {
        return torch.getattr("tensor")?.call1((batch,));
    }
    if is_instance(&builtins, &element, &builtins.getattr("str")?)?
        || is_instance(&builtins, &element, &builtins.getattr("bytes")?)?
    {
        return Ok(batch.clone());
    }

    let mapping = collections.getattr("Mapping")?;
    if is_instance(&builtins, &element, &mapping)? {
        let mutable = is_instance(&builtins, &element, &collections.getattr("MutableMapping")?)?;
        return mapping_collate(py, batch, &element, mutable);
    }

    if is_instance(&builtins, &element, &builtins.getattr("tuple")?)?
        && element.hasattr("_fields")?
    {
        let columns = collated_columns(py, batch, element.len()?)?;
        return element.get_type().call1(PyTuple::new(py, columns)?);
    }

    if is_instance(&builtins, &element, &collections.getattr("Sequence")?)? {
        return sequence_collate(
            py,
            batch,
            &element,
            &builtins,
            &collections.getattr("MutableSequence")?,
        );
    }

    if custom_registration_matches(py, &element, &builtins)? {
        return Err(PyTypeError::new_err(
            "hyperloader engine collation does not support custom default_collate \
             registrations; provide collate_fn",
        ));
    }
    Err(PyTypeError::new_err(format!(
        "default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}",
        element.get_type().repr()?
    )))
}

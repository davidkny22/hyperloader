//! Top-level default-collation dispatch.

use super::container::{collated_columns, mapping_collate, sequence_collate};
use super::kind::is_instance;
use super::tensor::tensor_collate;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

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

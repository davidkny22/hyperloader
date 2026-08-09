//! Tensor layout validation and stacking.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub(super) fn tensor_collate<'py>(
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

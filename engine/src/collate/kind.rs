//! Python type predicates used by collation dispatch.

use pyo3::prelude::*;

pub(super) fn is_instance(
    builtins: &Bound<'_, PyModule>,
    value: &Bound<'_, PyAny>,
    class: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    builtins
        .getattr("isinstance")?
        .call1((value, class))?
        .extract()
}

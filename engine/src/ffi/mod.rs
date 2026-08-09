//! Python extension registration is centralized in this module.

use pyo3::prelude::*;

use crate::rng;

/// Return the package version embedded in the native extension.
#[pyfunction]
pub(crate) fn package_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction(name = "_rng_block")]
fn rng_block(
    root_seed: u64,
    epoch: u64,
    coord: u64,
    draw_index: u32,
    stream_id: u32,
) -> (u32, u32, u32, u32) {
    let words = rng::block(root_seed, epoch, coord, draw_index, stream_id);
    (words[0], words[1], words[2], words[3])
}

#[pyfunction(name = "_sample_seed_words")]
fn sample_seed_words(root_seed: u64, epoch: u64, coord: u64) -> (u64, u64, (u32, u32, u32, u32)) {
    let (torch_seed, random_seed, numpy) = rng::sample_seed_words(root_seed, epoch, coord);
    (
        torch_seed,
        random_seed,
        (numpy[0], numpy[1], numpy[2], numpy[3]),
    )
}

/// Register the stable native functions exposed by the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(package_version, module)?)?;
    module.add_function(wrap_pyfunction!(rng_block, module)?)?;
    module.add_function(wrap_pyfunction!(sample_seed_words, module)?)?;
    Ok(())
}

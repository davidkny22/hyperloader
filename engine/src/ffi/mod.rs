//! Python extension registration is centralized in this module.

use crate::{collate, rng};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod process;
mod profile;
mod schedule;

/// Return the package version embedded in the native extension.
#[pyfunction]
pub(crate) fn package_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests;

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

#[pyfunction(name = "_rng_block_from_key")]
fn rng_block_from_key(
    key: u64,
    coord: u64,
    draw_index: u32,
    stream_id: u32,
) -> (u32, u32, u32, u32) {
    let words = rng::block_from_key(key, coord, draw_index, stream_id);
    (words[0], words[1], words[2], words[3])
}

#[pyfunction(name = "_sample_rng_context")]
fn sample_rng_context(root_seed: u64, epoch: u64, coord: u64) -> (u64, u64, u64) {
    let (torch_seed, key) = rng::sample_rng_context(root_seed, epoch, coord);
    (torch_seed, key, coord)
}

#[pyfunction(name = "_feistel_permute")]
fn feistel_permute(root_seed: u64, epoch: u64, domain: u64, position: u64) -> PyResult<u64> {
    rng::feistel_permute(root_seed, epoch, domain, position).ok_or_else(|| {
        PyValueError::new_err(
            "the Feistel permutation requires a domain of at least 131072 and a position inside it",
        )
    })
}

#[pyfunction(name = "_materialized_permutation")]
fn materialized_permutation(root_seed: u64, epoch: u64, domain: u32) -> PyResult<Vec<u32>> {
    rng::materialized_permutation(root_seed, epoch, domain).ok_or_else(|| {
        PyValueError::new_err("the materialized permutation requires a domain smaller than 131072")
    })
}

#[pyfunction(name = "_permutation_index")]
fn permutation_index(root_seed: u64, epoch: u64, domain: u64, position: u64) -> PyResult<u64> {
    rng::permutation_index(root_seed, epoch, domain, position)
        .ok_or_else(|| PyValueError::new_err("the permutation position must be inside its domain"))
}

#[pyfunction(name = "_rank_placements")]
#[pyo3(signature = (root_seed, epoch, dataset_len, batch_size, world_size, rank, drop_last=false, exact_count=false))]
#[expect(
    clippy::too_many_arguments,
    reason = "The private verification seam mirrors the eight contract inputs directly."
)]
fn rank_placements(
    root_seed: u64,
    epoch: u64,
    dataset_len: u64,
    batch_size: u64,
    world_size: u64,
    rank: u64,
    drop_last: bool,
    exact_count: bool,
) -> PyResult<Vec<(u64, u64)>> {
    let request = rng::PlacementRequest {
        root_seed,
        epoch,
        dataset_len,
        batch_size,
        world_size,
        rank,
        drop_last,
        exact_count,
    };
    rng::rank_placements(request)
        .map(|items| {
            items
                .into_iter()
                .map(|item| (item.position, item.index))
                .collect()
        })
        .map_err(|error| {
            PyValueError::new_err(format!("invalid placement configuration: {error:?}"))
        })
}

#[pyfunction(name = "_default_collate")]
fn default_collate<'py>(py: Python<'py>, batch: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    collate::default_collate(py, batch)
}

/// Register the stable native functions exposed by the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(package_version, module)?)?;
    module.add_function(wrap_pyfunction!(rng_block, module)?)?;
    module.add_function(wrap_pyfunction!(rng_block_from_key, module)?)?;
    module.add_function(wrap_pyfunction!(sample_rng_context, module)?)?;
    module.add_function(wrap_pyfunction!(feistel_permute, module)?)?;
    module.add_function(wrap_pyfunction!(materialized_permutation, module)?)?;
    module.add_function(wrap_pyfunction!(permutation_index, module)?)?;
    module.add_function(wrap_pyfunction!(rank_placements, module)?)?;
    module.add_function(wrap_pyfunction!(default_collate, module)?)?;
    process::register(module)?;
    module.add_class::<profile::PyCostProfile>()?;
    module.add_class::<schedule::PyStaticSchedule>()?;
    Ok(())
}

//! Counter-based random streams, permutations, and placement functions live in this module.

mod permutation;
mod philox;
mod placement;
mod streams;

pub use permutation::{
    FEISTEL_THRESHOLD, feistel_permute, materialized_permutation, permutation_index,
};
pub use philox::philox4x32_10;
pub use placement::{
    PlacedSample, PlacementError, PlacementRequest, elastic_batch_size, rank_placements,
};
pub use streams::{
    ACCESSOR_NUMPY_STREAM, ACCESSOR_RANDOM_STREAM, ACCESSOR_TORCH_STREAM, COLLATE_STREAM,
    SAMPLE_STREAM, STATE_NUMPY_STREAM, STATE_RANDOM_STREAM, block, key64, mt19937_state,
    sample_rng_states, sample_torch_seed, splitmix64,
};

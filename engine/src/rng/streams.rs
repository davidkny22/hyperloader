//! Stream-separated key and seed derivation.

use super::philox::philox4x32_10;

/// The stream used for sample-level global RNG installation.
pub const SAMPLE_STREAM: u32 = 0;
/// The stream used for batch-level collation RNG installation.
pub const COLLATE_STREAM: u32 = 1;
/// The stream used by the provided torch generator accessor.
pub const ACCESSOR_TORCH_STREAM: u32 = 4;
/// The stream used by the provided NumPy generator accessor.
pub const ACCESSOR_NUMPY_STREAM: u32 = 5;
/// The stream used by the provided Python random accessor.
pub const ACCESSOR_RANDOM_STREAM: u32 = 6;
/// The stream used by the installed Python random surface.
pub const STATE_RANDOM_STREAM: u32 = 7;
/// The stream used to key the installed NumPy Philox surface.
pub const STATE_NUMPY_STREAM: u32 = 8;

/// Apply the stateless SplitMix64 finalizer without a gamma increment.
pub const fn splitmix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    value
}

/// Derive the epoch-specific 64-bit key from the root seed.
pub const fn key64(root_seed: u64, epoch: u64) -> u64 {
    let epoch_tag = epoch.wrapping_shl(32) | 0x9E37;
    splitmix64(root_seed) ^ splitmix64(epoch_tag)
}

/// Derive one stream-separated 128-bit block for a sample coordinate.
pub const fn block(
    root_seed: u64,
    epoch: u64,
    coord: u64,
    draw_index: u32,
    stream_id: u32,
) -> [u32; 4] {
    let key = key64(root_seed, epoch);
    block_from_key(key, coord, draw_index, stream_id)
}

/// Derive one stream-separated block from an already resolved epoch key.
pub const fn block_from_key(key: u64, coord: u64, draw_index: u32, stream_id: u32) -> [u32; 4] {
    philox4x32_10(
        [coord as u32, draw_index, stream_id, (coord >> 32) as u32],
        [key as u32, (key >> 32) as u32],
    )
}

/// Return the CPU torch seed reserved for one sample.
pub const fn sample_torch_seed(root_seed: u64, epoch: u64, coord: u64) -> u64 {
    sample_rng_context(root_seed, epoch, coord).0
}

/// Return the CPU torch seed and the epoch key used by installed RNG surfaces.
pub const fn sample_rng_context(root_seed: u64, epoch: u64, coord: u64) -> (u64, u64) {
    let key = key64(root_seed, epoch);
    let globals = block_from_key(key, coord, 0, SAMPLE_STREAM);
    let seed = globals[0] as u64 | ((globals[1] as u64) << 32);
    (seed, key)
}

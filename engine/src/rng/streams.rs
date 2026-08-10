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
/// The stream used to synthesize the Python random MT19937 state.
pub const STATE_RANDOM_STREAM: u32 = 7;
/// The stream used to synthesize the NumPy legacy MT19937 state.
pub const STATE_NUMPY_STREAM: u32 = 8;

const MT19937_WORDS: usize = 624;

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
    philox4x32_10(
        [coord as u32, draw_index, stream_id, (coord >> 32) as u32],
        [key as u32, (key >> 32) as u32],
    )
}

/// Return the CPU torch seed reserved for one sample.
pub const fn sample_torch_seed(root_seed: u64, epoch: u64, coord: u64) -> u64 {
    let globals = block(root_seed, epoch, coord, 0, SAMPLE_STREAM);
    globals[0] as u64 | ((globals[1] as u64) << 32)
}

/// Synthesize one complete MT19937 state from its dedicated Philox stream.
pub fn mt19937_state(
    root_seed: u64,
    epoch: u64,
    coord: u64,
    stream_id: u32,
) -> [u32; MT19937_WORDS] {
    assert!(
        stream_id == STATE_RANDOM_STREAM || stream_id == STATE_NUMPY_STREAM,
        "MT19937 state requires a dedicated state stream"
    );
    let mut state = [0_u32; MT19937_WORDS];
    for draw_index in 0..156_u32 {
        let words = block(root_seed, epoch, coord, draw_index, stream_id);
        let start = draw_index as usize * 4;
        state[start..start + 4].copy_from_slice(&words);
    }
    repair_zero_state(&mut state, block(root_seed, epoch, coord, 156, stream_id));
    state
}

/// Return torch's seed and both whole legacy MT19937 states for one sample.
pub fn sample_rng_states(
    root_seed: u64,
    epoch: u64,
    coord: u64,
) -> (u64, [u32; MT19937_WORDS], [u32; MT19937_WORDS]) {
    (
        sample_torch_seed(root_seed, epoch, coord),
        mt19937_state(root_seed, epoch, coord, STATE_RANDOM_STREAM),
        mt19937_state(root_seed, epoch, coord, STATE_NUMPY_STREAM),
    )
}

fn repair_zero_state(state: &mut [u32; MT19937_WORDS], regeneration: [u32; 4]) {
    if state.iter().all(|word| *word == 0) {
        state[..4].copy_from_slice(&regeneration);
    }
}

#[cfg(test)]
mod tests;

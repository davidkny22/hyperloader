//! Counter-based random streams, permutations, and placement functions live in this module.

const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;

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

const fn multiply_words(left: u32, right: u32) -> (u32, u32) {
    let product = (left as u64) * (right as u64);
    (product as u32, (product >> 32) as u32)
}

/// Evaluate Philox4x32-10 for one four-word counter and two-word key.
pub const fn philox4x32_10(mut counter: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    let mut round = 0;
    while round < 10 {
        let (lo0, hi0) = multiply_words(PHILOX_M0, counter[0]);
        let (lo1, hi1) = multiply_words(PHILOX_M1, counter[2]);
        counter = [
            hi1 ^ counter[1] ^ key[0],
            lo1,
            hi0 ^ counter[3] ^ key[1],
            lo0,
        ];
        if round != 9 {
            key[0] = key[0].wrapping_add(PHILOX_W0);
            key[1] = key[1].wrapping_add(PHILOX_W1);
        }
        round += 1;
    }
    counter
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

/// Return the two global seeds and four NumPy words reserved for one sample.
pub const fn sample_seed_words(root_seed: u64, epoch: u64, coord: u64) -> (u64, u64, [u32; 4]) {
    let globals = block(root_seed, epoch, coord, 0, SAMPLE_STREAM);
    let numpy = block(root_seed, epoch, coord, 1, SAMPLE_STREAM);
    let torch_seed = globals[0] as u64 | ((globals[1] as u64) << 32);
    let random_seed = globals[2] as u64 | ((globals[3] as u64) << 32);
    (torch_seed, random_seed, numpy)
}

#[cfg(test)]
mod tests {
    use super::{SAMPLE_STREAM, block, key64, philox4x32_10, sample_seed_words, splitmix64};

    #[test]
    fn random123_zero_vector_matches() {
        assert_eq!(
            philox4x32_10([0; 4], [0; 2]),
            [0x6627_E8D5, 0xE169_C58D, 0xBC57_AC4C, 0x9B00_DBD8]
        );
    }

    #[test]
    fn splitmix_is_a_bare_finalizer() {
        assert_eq!(splitmix64(0), 0);
        assert_ne!(splitmix64(1), splitmix64(2));
        assert_ne!(key64(7, 0), key64(7, 1));
    }

    #[test]
    fn sample_seed_words_use_the_two_reserved_draws() {
        let (torch_seed, random_seed, numpy) = sample_seed_words(11, 3, 29);
        let globals = block(11, 3, 29, 0, SAMPLE_STREAM);

        assert_eq!(torch_seed, globals[0] as u64 | ((globals[1] as u64) << 32));
        assert_eq!(random_seed, globals[2] as u64 | ((globals[3] as u64) << 32));
        assert_eq!(numpy, block(11, 3, 29, 1, SAMPLE_STREAM));
        assert_ne!(numpy, block(11, 3, 29, 2, SAMPLE_STREAM));
    }
}

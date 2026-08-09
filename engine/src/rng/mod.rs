//! Counter-based random streams, permutations, and placement functions live in this module.

const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;
const PERM_TAG: u64 = 2;
const PERM_ROUND_TAG: u32 = 3;

/// The first domain size that uses the stateless Feistel permutation.
pub const FEISTEL_THRESHOLD: u64 = 1 << 17;

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

const fn low_mask(width: u32) -> u32 {
    if width == 32 {
        u32::MAX
    } else {
        (1_u32 << width) - 1
    }
}

const fn permutation_key(root_seed: u64, epoch: u64) -> [u32; 2] {
    let folded = splitmix64(key64(root_seed, epoch) ^ ((PERM_TAG << 1) | 1));
    [folded as u32, (folded >> 32) as u32]
}

const fn feistel_cipher(value: u64, bits: u32, key: [u32; 2]) -> u64 {
    let lower_width = bits / 2;
    let mut high_width = bits - lower_width;
    let mut low_width = lower_width;
    let mut high = (value >> lower_width) as u32;
    let mut low = value as u32 & low_mask(lower_width);
    let mut round = 0_u32;

    while round < 8 {
        let function =
            philox4x32_10([low, round, PERM_ROUND_TAG, 0], key)[0] & low_mask(high_width);
        let next_high = low;
        let next_low = high.wrapping_add(function) & low_mask(high_width);
        high = next_high;
        low = next_low;
        let next_high_width = low_width;
        low_width = high_width;
        high_width = next_high_width;
        round += 1;
    }

    ((high as u64) << lower_width) | low as u64
}

/// Permute one position in a large domain using cycle-walked unbalanced Feistel.
pub fn feistel_permute(root_seed: u64, epoch: u64, domain: u64, position: u64) -> Option<u64> {
    if domain < FEISTEL_THRESHOLD || position >= domain {
        return None;
    }
    let bits = 64 - (domain - 1).leading_zeros();
    let key = permutation_key(root_seed, epoch);
    let mut candidate = position;
    loop {
        candidate = feistel_cipher(candidate, bits, key);
        if candidate < domain {
            return Some(candidate);
        }
    }
}

fn materialized_permutation_with_draws(
    root_seed: u64,
    epoch: u64,
    domain: u32,
) -> Option<(Vec<u32>, u32)> {
    if domain as u64 >= FEISTEL_THRESHOLD {
        return None;
    }
    let key = permutation_key(root_seed, epoch);
    let mut permutation: Vec<u32> = (0..domain).collect();
    let mut draw_ordinal = 0_u32;
    let mut upper = domain;
    while upper > 1 {
        let modulus = upper as u64;
        let limit = (1_u64 << 32) - ((1_u64 << 32) % modulus);
        let selected = loop {
            let word = philox4x32_10([draw_ordinal, 8, PERM_ROUND_TAG, 0], key)[0];
            draw_ordinal = draw_ordinal
                .checked_add(1)
                .expect("small-domain permutation draw ordinal cannot overflow");
            if (word as u64) < limit {
                break (word as u64 % modulus) as usize;
            }
        };
        permutation.swap((upper - 1) as usize, selected);
        upper -= 1;
    }
    Some((permutation, draw_ordinal))
}

/// Materialize the exact-uniform Fisher-Yates permutation for a small domain.
pub fn materialized_permutation(root_seed: u64, epoch: u64, domain: u32) -> Option<Vec<u32>> {
    materialized_permutation_with_draws(root_seed, epoch, domain)
        .map(|(permutation, _)| permutation)
}

/// Return the permutation index for either side of the frozen regime threshold.
pub fn permutation_index(root_seed: u64, epoch: u64, domain: u64, position: u64) -> Option<u64> {
    if position >= domain {
        return None;
    }
    if domain < FEISTEL_THRESHOLD {
        let domain = u32::try_from(domain).ok()?;
        return materialized_permutation(root_seed, epoch, domain)
            .map(|permutation| permutation[position as usize] as u64);
    }
    feistel_permute(root_seed, epoch, domain, position)
}

#[cfg(test)]
mod tests {
    use super::{
        FEISTEL_THRESHOLD, SAMPLE_STREAM, block, feistel_permute, key64,
        materialized_permutation_with_draws, permutation_index, philox4x32_10, sample_seed_words,
        splitmix64,
    };

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

    #[test]
    fn feistel_threshold_domain_is_bijective() {
        let mut seen = vec![false; FEISTEL_THRESHOLD as usize];
        for position in 0..FEISTEL_THRESHOLD {
            let output = feistel_permute(17, 4, FEISTEL_THRESHOLD, position).unwrap();
            assert!(!seen[output as usize]);
            seen[output as usize] = true;
        }
        assert!(seen.into_iter().all(|value| value));
    }

    #[test]
    fn feistel_rejects_small_domains_and_invalid_positions() {
        assert_eq!(feistel_permute(0, 0, FEISTEL_THRESHOLD - 1, 0), None);
        assert_eq!(
            feistel_permute(0, 0, FEISTEL_THRESHOLD, FEISTEL_THRESHOLD),
            None
        );
    }

    #[test]
    fn materialized_regime_is_bijective_at_its_largest_domain() {
        let domain = (FEISTEL_THRESHOLD - 1) as u32;
        let (permutation, draws) = materialized_permutation_with_draws(0, 0, domain).unwrap();
        let mut sorted = permutation;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..domain).collect::<Vec<_>>());
        assert!(draws >= domain - 1);
    }

    #[test]
    fn unified_permutation_switches_at_the_frozen_threshold() {
        assert_eq!(permutation_index(3, 5, 1, 0), Some(0));
        assert!(permutation_index(3, 5, FEISTEL_THRESHOLD - 1, 0).is_some());
        assert!(permutation_index(3, 5, FEISTEL_THRESHOLD, 0).is_some());
        assert_eq!(permutation_index(3, 5, 3, 3), None);
    }
}

//! Exact small-domain and stateless large-domain permutations.

use super::philox::philox4x32_10;
use super::streams::{key64, splitmix64};

const PERM_TAG: u64 = 2;
const PERM_ROUND_TAG: u32 = 3;

/// The first domain size that uses the stateless Feistel permutation.
pub const FEISTEL_THRESHOLD: u64 = 1 << 17;

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
mod tests;

//! Philox4x32-10 primitive.

const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;

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

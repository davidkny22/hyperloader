use _hyperloader::rng::{
    FEISTEL_THRESHOLD, SAMPLE_STREAM, STATE_NUMPY_STREAM, STATE_RANDOM_STREAM, block,
    block_from_key, feistel_permute, key64, permutation_index, philox4x32_10, sample_rng_context,
    sample_torch_seed, splitmix64,
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
fn sample_rng_material_uses_separated_streams() {
    let (torch_seed, key) = sample_rng_context(11, 3, 29);
    let globals = block(11, 3, 29, 0, SAMPLE_STREAM);
    assert_eq!(torch_seed, globals[0] as u64 | ((globals[1] as u64) << 32));
    assert_eq!(torch_seed, sample_torch_seed(11, 3, 29));
    assert_eq!(key, key64(11, 3));
    assert_eq!(
        block_from_key(key, 29, 0, STATE_RANDOM_STREAM),
        block(11, 3, 29, 0, STATE_RANDOM_STREAM)
    );
    assert_ne!(
        block_from_key(key, 29, 0, STATE_RANDOM_STREAM),
        block_from_key(key, 29, 0, STATE_NUMPY_STREAM)
    );
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
fn unified_permutation_switches_at_the_frozen_threshold() {
    assert_eq!(permutation_index(3, 5, 1, 0), Some(0));
    assert!(permutation_index(3, 5, FEISTEL_THRESHOLD - 1, 0).is_some());
    assert!(permutation_index(3, 5, FEISTEL_THRESHOLD, 0).is_some());
    assert_eq!(permutation_index(3, 5, 3, 3), None);
}

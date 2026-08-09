use _hyperloader::rng::{
    FEISTEL_THRESHOLD, SAMPLE_STREAM, block, feistel_permute, key64, permutation_index,
    philox4x32_10, sample_seed_words, splitmix64,
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
fn unified_permutation_switches_at_the_frozen_threshold() {
    assert_eq!(permutation_index(3, 5, 1, 0), Some(0));
    assert!(permutation_index(3, 5, FEISTEL_THRESHOLD - 1, 0).is_some());
    assert!(permutation_index(3, 5, FEISTEL_THRESHOLD, 0).is_some());
    assert_eq!(permutation_index(3, 5, 3, 3), None);
}

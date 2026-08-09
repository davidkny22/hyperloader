use super::{FEISTEL_THRESHOLD, materialized_permutation_with_draws};

#[test]
fn materialized_regime_is_bijective_at_its_largest_domain() {
    let domain = (FEISTEL_THRESHOLD - 1) as u32;
    let (permutation, draws) = materialized_permutation_with_draws(0, 0, domain).unwrap();
    let mut sorted = permutation;
    sorted.sort_unstable();
    assert_eq!(sorted, (0..domain).collect::<Vec<_>>());
    assert!(draws >= domain - 1);
}

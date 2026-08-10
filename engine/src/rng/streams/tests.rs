use super::{MT19937_WORDS, repair_zero_state};

#[test]
fn all_zero_state_uses_the_regeneration_block() {
    let mut state = [0_u32; MT19937_WORDS];
    repair_zero_state(&mut state, [3, 5, 7, 11]);

    assert_eq!(&state[..4], &[3, 5, 7, 11]);
    assert!(state[4..].iter().all(|word| *word == 0));
}

#[test]
fn nonzero_state_is_not_replaced() {
    let mut state = [0_u32; MT19937_WORDS];
    state[17] = 13;
    repair_zero_state(&mut state, [3, 5, 7, 11]);

    assert_eq!(state[17], 13);
    assert!(state[..4].iter().all(|word| *word == 0));
}

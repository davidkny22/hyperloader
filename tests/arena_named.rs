use _hyperloader::arena::RegionToken;

#[test]
fn random_tokens_are_not_reused() {
    let first = RegionToken::random().expect("first token");
    let second = RegionToken::random().expect("second token");
    assert_ne!(first, second);
}

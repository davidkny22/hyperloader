use super::package_version;

#[test]
fn package_version_matches_manifest() {
    assert_eq!(package_version(), env!("CARGO_PKG_VERSION"));
}

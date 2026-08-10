use super::CostProfile;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn direct_profile_applies_the_configured_ema() {
    let mut profile = CostProfile::new(4, 32, 0.3).expect("valid profile");
    profile.observe(2, 100).expect("first observation");
    profile.observe(2, 200).expect("second observation");

    assert_eq!(profile.estimate(2).expect("inside domain"), Some(130.0));
    assert!(!profile.is_degraded());
    assert_eq!(profile.payload_bytes(), 32);
}

#[test]
fn budget_clamp_degrades_to_stable_hash_buckets() {
    let mut profile = CostProfile::new(100, 16, 0.3).expect("valid profile");
    profile.observe(7, 100).expect("observation");

    let first = profile.estimate(7).expect("inside domain");
    let second = profile.estimate(7).expect("inside domain");
    assert_eq!(first, second);
    assert!(profile.is_degraded());
    assert_eq!(profile.payload_bytes(), 16);
}

#[test]
fn statistics_use_only_populated_estimates() {
    let mut profile = CostProfile::new(4, 32, 0.3).expect("valid profile");
    for (position, cost) in [10, 20, 30, 1_000].into_iter().enumerate() {
        profile.observe(position as u64, cost).expect("observation");
    }

    let statistics = profile.statistics().expect("populated statistics");
    assert_eq!(statistics.mean_ns, 265.0);
    assert_eq!(statistics.p999_ns, 1_000.0);
    assert_eq!(statistics.populated, 4);
}

#[test]
fn binary_profile_round_trips_and_rejects_changed_inputs() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    let directory = std::env::temp_dir().join(format!("hyperloader-profile-{unique}"));
    let path = directory.join("costs.bin");
    let mut profile = CostProfile::new(3, 24, 0.3).expect("valid profile");
    profile.observe(1, 77).expect("observation");
    profile.save(&path).expect("profile saved");

    let loaded = CostProfile::load(&path, 3, 24, 0.3).expect("profile loaded");
    assert_eq!(loaded.estimate(1).expect("inside domain"), Some(77.0));
    assert!(CostProfile::load(&path, 4, 24, 0.3).is_err());
    assert!(CostProfile::load(&path, 3, 8, 0.3).is_err());
    assert!(CostProfile::load(&path, 3, 24, 0.4).is_err());

    fs::remove_file(&path).expect("profile removed");
    fs::remove_dir(&directory).expect("directory removed");
}

#[test]
fn invalid_observations_and_domains_are_rejected() {
    assert!(CostProfile::new(1, 8, 0.0).is_err());
    let mut profile = CostProfile::new(1, 8, 0.3).expect("valid profile");
    assert!(profile.observe(1, 10).is_err());
    assert!(profile.observe(0, 0).is_err());
    assert!(profile.estimate(1).is_err());
}

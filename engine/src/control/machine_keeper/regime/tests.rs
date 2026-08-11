use super::needs_activity;

#[test]
fn zero_powered_down_entries_keep_activity_parked() {
    assert!(!needs_activity(Some(0)));
}

#[test]
fn powered_down_entries_request_activity() {
    assert!(needs_activity(Some(1)));
}

#[test]
fn unavailable_counters_use_the_conservative_fallback() {
    assert!(needs_activity(None));
}

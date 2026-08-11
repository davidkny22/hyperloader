use super::ActivityProbe;

#[test]
fn first_counter_boundary_only_primes_the_probe() {
    let mut probe = ActivityProbe::new();

    assert!(!probe.observe(Some(7)));
}

#[test]
fn complete_zero_entry_window_keeps_activity_parked() {
    let mut probe = ActivityProbe::new();

    assert!(!probe.observe(Some(0)));
    assert!(!probe.observe(Some(0)));
}

#[test]
fn complete_powered_down_window_requests_activity() {
    let mut probe = ActivityProbe::new();

    assert!(!probe.observe(Some(0)));
    assert!(probe.observe(Some(1)));
}

#[test]
fn unavailable_counters_use_the_conservative_fallback() {
    assert!(ActivityProbe::new().observe(None));
}

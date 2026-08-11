use super::PeriodicPulse;

#[test]
fn periodic_deadline_advances_without_rebasing() {
    let mut pulse = PeriodicPulse::new(1);

    #[cfg(target_os = "linux")]
    {
        let first = pulse.next_start_ns;
        pulse.wait_next();
        assert_eq!(pulse.next_start_ns, first + 1);
    }

    #[cfg(not(target_os = "linux"))]
    {
        let first = pulse.next_start;
        pulse.wait_next();
        assert_eq!(pulse.next_start.duration_since(first), pulse.period);
    }
}

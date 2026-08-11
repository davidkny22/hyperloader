use super::PeriodicPulse;
use std::time::Duration;

#[test]
fn periodic_deadline_skips_missed_intervals_on_its_original_grid() {
    let period_ns = 1_000_000;
    let mut pulse = PeriodicPulse::new(period_ns);

    #[cfg(target_os = "linux")]
    {
        let first = pulse.next_start_ns;
        std::thread::sleep(Duration::from_millis(3));
        pulse.wait_next();
        assert!(pulse.next_start_ns > first + period_ns);
        assert_eq!((pulse.next_start_ns - first) % period_ns, 0);
    }

    #[cfg(not(target_os = "linux"))]
    {
        let first = pulse.next_start;
        std::thread::sleep(Duration::from_millis(3));
        pulse.wait_next();
        let elapsed = pulse.next_start.duration_since(first);
        assert!(elapsed > pulse.period);
        assert_eq!(elapsed.as_nanos() % pulse.period.as_nanos(), 0);
    }
}

#[test]
fn active_window_expires_when_the_thread_misses_its_period() {
    let pulse = PeriodicPulse::new(1_000_000);
    std::thread::sleep(Duration::from_millis(2));

    #[cfg(target_os = "linux")]
    {
        assert!(pulse.active_until(100_000) < super::monotonic_ns());
    }

    #[cfg(not(target_os = "linux"))]
    {
        assert!(pulse.active_until(100_000) < std::time::Instant::now());
    }
}

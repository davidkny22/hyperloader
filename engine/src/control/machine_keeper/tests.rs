use super::tuner::DutyTuner;
use super::{DUTY_SCALE, MachineKeeper};
use std::thread;
use std::time::Duration;

#[test]
fn tuner_restores_the_lowest_zero_entry_duty() {
    let mut tuner = DutyTuner::new(20_000, 50_000);

    assert_eq!(tuner.observe(1), 30_000);
    assert_eq!(tuner.observe(0), 20_000);
    assert_eq!(tuner.observe(1), 30_000);
    assert_eq!(tuner.observe(0), 30_000);
}

#[test]
fn tuner_does_not_undercut_the_calibrated_warm_duty() {
    let mut tuner = DutyTuner::new(50_000, 50_000);

    assert_eq!(tuner.observe(0), 50_000);
    assert_eq!(tuner.observe(0), 50_000);
}

#[test]
fn tuner_never_exceeds_the_configured_cap() {
    let mut tuner = DutyTuner::new(30_000, 50_000);

    assert_eq!(tuner.observe(1), 40_000);
    assert_eq!(tuner.observe(1), 50_000);
    assert_eq!(tuner.observe(1), 50_000);
    assert_eq!(DUTY_SCALE, 1_000_000);
}

#[test]
fn tuner_reopens_after_a_late_powered_down_entry() {
    let mut tuner = DutyTuner::new(20_000, 50_000);

    assert_eq!(tuner.observe(0), 20_000);
    assert_eq!(tuner.observe(0), 20_000);
    assert_eq!(tuner.observe(0), 20_000);
    assert_eq!(tuner.observe(1), 30_000);
    assert_eq!(tuner.observe(1), 40_000);
}

#[test]
fn gap_gate_parks_and_activates_the_native_thread() {
    let cpu = current_cpu();
    let mut keeper = MachineKeeper::new(vec![cpu], 0.05, 0.05, 2_000_000)
        .expect("machine keeper should start on the current CPU");

    keeper.observe_gap(1_999_999);
    assert_eq!(keeper.duty(), 0.0);
    keeper.observe_gap(2_000_000);
    thread::sleep(Duration::from_millis(2));
    assert!(keeper.duty() > 0.0);
    assert!(keeper.duty() <= 0.05);
    keeper.park();
    assert_eq!(keeper.duty(), 0.0);
    keeper.close();
}

#[test]
fn deferred_park_spans_rollover_and_expires_without_consumption() {
    let cpu = current_cpu();
    let mut keeper = MachineKeeper::new(vec![cpu], 0.05, 0.05, 2_000_000)
        .expect("machine keeper should start on the current CPU");

    keeper.observe_gap(2_000_000);
    keeper.defer_park(20_000_000);
    keeper.observe_gap(2_000_000);
    thread::sleep(Duration::from_millis(30));
    assert!(keeper.duty() > 0.0);

    keeper.defer_park(1_000_000);
    thread::sleep(Duration::from_millis(30));
    assert_eq!(keeper.duty(), 0.0);
    keeper.close();
}

#[cfg(target_os = "linux")]
fn current_cpu() -> usize {
    // SAFETY: sched_getcpu takes no pointers and reports the calling thread's CPU.
    unsafe { libc::sched_getcpu().max(0) as usize }
}

#[cfg(not(target_os = "linux"))]
fn current_cpu() -> usize {
    0
}

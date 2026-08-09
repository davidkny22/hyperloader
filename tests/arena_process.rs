use _hyperloader::arena::{ProcessObservation, ProcessObserver, SystemProcessObserver};
use std::process::Command;
use std::time::Duration;

#[test]
fn system_observer_reports_a_stable_current_identity() {
    let observer = SystemProcessObserver;
    let first = observer.current_identity().expect("first current identity");
    let second = observer
        .current_identity()
        .expect("second current identity");
    assert_eq!(first, second);
    assert!(!first.boot_id.is_empty());
    #[cfg(windows)]
    assert_ne!(first.boot_id, "windows-boot-id-unavailable");
    assert_ne!(first.proc_start, 0);
    assert_eq!(
        observer.observe(std::process::id()),
        ProcessObservation::Live(first)
    );
}

#[test]
fn impossible_pid_is_reported_missing() {
    assert_eq!(
        SystemProcessObserver.observe(u32::MAX),
        ProcessObservation::Missing
    );
}

#[test]
fn exited_child_is_proven_missing() {
    let mut child = Command::new(std::env::current_exe().expect("current test executable"))
        .args(["--exact", "long_lived_process_helper", "--ignored"])
        .spawn()
        .expect("spawn liveness child");
    let mut live = false;
    for _ in 0..100 {
        if matches!(
            SystemProcessObserver.observe(child.id()),
            ProcessObservation::Live(_)
        ) {
            live = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    child.kill().expect("terminate liveness child");
    child.wait().expect("reap liveness child");
    assert!(live);
    assert_eq!(
        SystemProcessObserver.observe(child.id()),
        ProcessObservation::Missing
    );
}

#[test]
#[ignore = "launched by exited_child_is_proven_missing"]
fn long_lived_process_helper() {
    std::thread::park_timeout(Duration::from_secs(30));
}

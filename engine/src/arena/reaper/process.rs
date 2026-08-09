//! Process identity observations used to guard orphan cleanup.

#[cfg(target_os = "linux")]
#[path = "process_linux.rs"]
mod platform;
#[cfg(target_os = "macos")]
#[path = "process_macos.rs"]
mod platform;
#[cfg(windows)]
#[path = "process_windows.rs"]
mod platform;

/// Boot and process-start identity recorded for one creator PID.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessIdentity {
    /// Platform boot identifier.
    pub boot_id: String,
    /// Process start time in platform-normalized units.
    pub proc_start: u64,
}

/// A conservative observation of one process identifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProcessObservation {
    /// The operating system proves that the PID is absent or exited.
    Missing,
    /// A live process exposed its complete identity.
    Live(ProcessIdentity),
    /// Permission, transient failure, or unsupported state prevents proof.
    Ambiguous(String),
}

/// Process observation seam used by the system reaper and deterministic tests.
pub trait ProcessObserver {
    /// Observe the calling process for a new ownership record.
    fn current_identity(&self) -> Result<ProcessIdentity, String>;
    /// Observe one recorded PID without turning uncertainty into death.
    fn observe(&self, pid: u32) -> ProcessObservation;
}

/// Operating-system process observer.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemProcessObserver;

impl ProcessObserver for SystemProcessObserver {
    fn current_identity(&self) -> Result<ProcessIdentity, String> {
        platform::current_identity()
    }

    fn observe(&self, pid: u32) -> ProcessObservation {
        platform::observe(pid)
    }
}

#[cfg(test)]
mod tests {
    use super::{ProcessObservation, ProcessObserver, SystemProcessObserver};
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
            .args([
                "--exact",
                "arena::reaper::process::tests::long_lived_process_helper",
                "--ignored",
            ])
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
}

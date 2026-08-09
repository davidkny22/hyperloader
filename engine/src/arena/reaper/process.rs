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

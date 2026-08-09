//! macOS process identity through the supported sysinfo platform backend.

use super::{ProcessIdentity, ProcessObservation};
use std::io;
use sysinfo::{Pid, ProcessesToUpdate, System};

pub(super) fn current_identity() -> Result<ProcessIdentity, String> {
    match observe(std::process::id()) {
        ProcessObservation::Live(identity) => Ok(identity),
        ProcessObservation::Missing => Err("calling process is absent".to_owned()),
        ProcessObservation::Ambiguous(detail) => Err(detail),
    }
}

pub(super) fn observe(pid: u32) -> ProcessObservation {
    let pid = Pid::from_u32(pid);
    let mut system = System::new();
    system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    let Some(process) = system.process(pid) else {
        // SAFETY: signal zero performs a liveness and permission check without sending a signal.
        return if unsafe { libc::kill(pid.as_u32() as libc::pid_t, 0) } == 0 {
            ProcessObservation::Ambiguous("macOS process identity is unreadable".to_owned())
        } else {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::ESRCH) {
                ProcessObservation::Missing
            } else {
                ProcessObservation::Ambiguous(format!(
                    "macOS process liveness is ambiguous: {error}"
                ))
            }
        };
    };
    ProcessObservation::Live(ProcessIdentity {
        boot_id: format!("boot-epoch-{}", System::boot_time()),
        proc_start: process.start_time(),
    })
}

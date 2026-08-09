//! Linux process identity from procfs boot and start records.

use super::{ProcessIdentity, ProcessObservation};
use std::fs;
use std::io;

pub(super) fn current_identity() -> Result<ProcessIdentity, String> {
    match observe(std::process::id()) {
        ProcessObservation::Live(identity) => Ok(identity),
        ProcessObservation::Missing => Err("calling process is absent from procfs".to_owned()),
        ProcessObservation::Ambiguous(detail) => Err(detail),
    }
}

pub(super) fn observe(pid: u32) -> ProcessObservation {
    let boot_id = match fs::read_to_string("/proc/sys/kernel/random/boot_id") {
        Ok(value) if !value.trim().is_empty() => value.trim().to_owned(),
        Ok(_) => return ProcessObservation::Ambiguous("Linux boot ID is empty".to_owned()),
        Err(error) => {
            return ProcessObservation::Ambiguous(format!("Linux boot ID is unreadable: {error}"));
        }
    };
    let stat_path = format!("/proc/{pid}/stat");
    let stat = match fs::read_to_string(stat_path) {
        Ok(value) => value,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return ProcessObservation::Missing;
        }
        Err(error) => {
            return ProcessObservation::Ambiguous(format!(
                "Linux process identity is unreadable: {error}"
            ));
        }
    };
    let Some(command_end) = stat.rfind(") ") else {
        return ProcessObservation::Ambiguous("Linux process stat is malformed".to_owned());
    };
    let fields: Vec<&str> = stat[command_end + 2..].split_whitespace().collect();
    if fields.first() == Some(&"Z") {
        return ProcessObservation::Missing;
    }
    let Some(proc_start) = fields.get(19).and_then(|value| value.parse::<u64>().ok()) else {
        return ProcessObservation::Ambiguous("Linux process start time is malformed".to_owned());
    };
    ProcessObservation::Live(ProcessIdentity {
        boot_id,
        proc_start,
    })
}

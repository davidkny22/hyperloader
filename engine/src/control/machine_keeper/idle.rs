//! Linux cpuidle entry-counter discovery and sampling.

use std::path::PathBuf;

pub(super) struct IdleEntryMonitor {
    paths: Vec<PathBuf>,
    previous: u64,
}

impl IdleEntryMonitor {
    pub(super) fn discover(cpus: &[usize]) -> Option<Self> {
        let paths = powered_down_paths(cpus);
        if paths.is_empty() {
            return None;
        }
        let previous = read_total(&paths)?;
        Some(Self { paths, previous })
    }

    pub(super) fn reset(&mut self) {
        if let Some(total) = read_total(&self.paths) {
            self.previous = total;
        }
    }

    pub(super) fn delta(&mut self) -> Option<u64> {
        let total = read_total(&self.paths)?;
        let delta = total.saturating_sub(self.previous);
        self.previous = total;
        Some(delta)
    }
}

#[cfg(target_os = "linux")]
fn powered_down_paths(cpus: &[usize]) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for cpu in cpus {
        let root = PathBuf::from(format!("/sys/devices/system/cpu/cpu{cpu}/cpuidle"));
        let Ok(states) = std::fs::read_dir(root) else {
            continue;
        };
        for state in states.flatten() {
            let name = state.file_name();
            let name = name.to_string_lossy();
            let Some(index) = name
                .strip_prefix("state")
                .and_then(|raw| raw.parse::<u32>().ok())
            else {
                continue;
            };
            if index > 0 && state.path().join("usage").is_file() {
                paths.push(state.path().join("usage"));
            }
        }
    }
    paths.sort();
    paths
}

#[cfg(not(target_os = "linux"))]
fn powered_down_paths(_cpus: &[usize]) -> Vec<PathBuf> {
    Vec::new()
}

fn read_total(paths: &[PathBuf]) -> Option<u64> {
    paths.iter().try_fold(0_u64, |total, path| {
        let value = std::fs::read_to_string(path).ok()?;
        Some(total.saturating_add(value.trim().parse::<u64>().ok()?))
    })
}

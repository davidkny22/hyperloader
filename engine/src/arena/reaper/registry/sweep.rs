//! Guarded orphan classification and cleanup.

use super::model::{RegistryEntry, RegistryError, RegistryIssue};
use super::store::RegionRegistry;
use crate::arena::RegionError;
use crate::arena::named::unlink_registered;
use crate::arena::reaper::{ProcessObservation, ProcessObserver, SystemProcessObserver};

/// The reason a sweep retained or removed one ownership record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SweepOutcome {
    /// The recorded process still owns the identity.
    Live,
    /// The recorded process is absent.
    Dead,
    /// The PID now belongs to another process or boot.
    Reused,
    /// The observation or unlink operation could not prove safety.
    Ambiguous,
}

/// One audit action emitted by a guarded sweep.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SweepAction {
    /// Region name from the validated registry record.
    pub name: String,
    /// Guard decision.
    pub outcome: SweepOutcome,
    /// Concise observation detail.
    pub detail: String,
}

/// Complete result of one guarded sweep and compaction attempt.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SweepReport {
    /// Records retained as live.
    pub live: usize,
    /// Records removed after death or reuse was proven.
    pub removed: usize,
    /// Records retained because safety could not be proven.
    pub ambiguous: usize,
    /// Per-record audit actions.
    pub actions: Vec<SweepAction>,
    /// Registry corruption that suppressed all destructive work.
    pub registry_issues: Vec<RegistryIssue>,
}

impl RegionRegistry {
    /// Sweep stale ownership records using the system process observer.
    pub fn sweep(&self) -> Result<SweepReport, RegistryError> {
        self.sweep_with(&SystemProcessObserver)
    }

    pub(super) fn sweep_with<O: ProcessObserver>(
        &self,
        observer: &O,
    ) -> Result<SweepReport, RegistryError> {
        let mut report = SweepReport::default();
        let snapshot = self.retain(|entry| {
            let (keep, outcome, detail) = match observer.observe(entry.pid) {
                ProcessObservation::Missing => match unlink_entry(entry) {
                    Ok(()) => (false, SweepOutcome::Dead, "process is absent".to_owned()),
                    Err(detail) => (true, SweepOutcome::Ambiguous, detail),
                },
                ProcessObservation::Live(identity)
                    if identity.boot_id == entry.boot_id
                        && identity.proc_start == entry.proc_start =>
                {
                    (
                        true,
                        SweepOutcome::Live,
                        "process identity matches".to_owned(),
                    )
                }
                ProcessObservation::Live(_) => match unlink_entry(entry) {
                    Ok(()) => (
                        false,
                        SweepOutcome::Reused,
                        "process identity differs".to_owned(),
                    ),
                    Err(detail) => (true, SweepOutcome::Ambiguous, detail),
                },
                ProcessObservation::Ambiguous(detail) => (true, SweepOutcome::Ambiguous, detail),
            };
            match outcome {
                SweepOutcome::Live => report.live += 1,
                SweepOutcome::Dead | SweepOutcome::Reused => report.removed += 1,
                SweepOutcome::Ambiguous => report.ambiguous += 1,
            }
            report.actions.push(SweepAction {
                name: entry.name.clone(),
                outcome,
                detail,
            });
            keep
        })?;
        if !snapshot.issues.is_empty() {
            report.ambiguous = snapshot.entries.len();
            report.registry_issues = snapshot.issues;
        }
        Ok(report)
    }
}

fn unlink_entry(entry: &RegistryEntry) -> Result<(), String> {
    let name = entry
        .validated_name()
        .ok_or_else(|| "registry identity is malformed".to_owned())?;
    match unlink_registered(&name) {
        Ok(()) | Err(RegionError::NotFound(_)) => Ok(()),
        Err(error) => Err(format!("region unlink remained ambiguous: {error}")),
    }
}

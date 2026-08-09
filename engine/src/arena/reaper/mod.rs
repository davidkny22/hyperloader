//! Crash-safe registry and guarded orphan cleanup for named arena regions.

mod process;
mod registry;

pub use process::{ProcessIdentity, ProcessObservation, ProcessObserver, SystemProcessObserver};
pub use registry::{
    RegionRegistry, RegistryEntry, RegistryError, RegistryIssue, RegistrySnapshot, SweepAction,
    SweepOutcome, SweepReport,
};

//! Crash-safe region ownership registry and guarded orphan cleanup.

mod cache;
mod format;
mod model;
mod store;
mod sweep;

pub use model::{RegistryEntry, RegistryError, RegistryIssue, RegistrySnapshot};
pub use store::RegionRegistry;
pub use sweep::{SweepAction, SweepOutcome, SweepReport};

#[cfg(test)]
use store::private_options;

#[cfg(test)]
mod tests;

//! Crash-safe registry and guarded orphan cleanup for named arena regions.

mod registry;

pub use registry::{RegionRegistry, RegistryEntry, RegistryError, RegistryIssue, RegistrySnapshot};

//! Shared-memory regions, slots, and delivery-view lifetimes live in this module.

mod named;
mod reaper;

pub use named::{NamedRegion, RegionError, RegionName, RegionToken};
pub use reaper::{RegionRegistry, RegistryEntry, RegistryError, RegistryIssue, RegistrySnapshot};

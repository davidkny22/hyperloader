//! Shared-memory regions, slots, and delivery-view lifetimes live in this module.

mod named;

pub use named::{NamedRegion, RegionError, RegionName, RegionToken};

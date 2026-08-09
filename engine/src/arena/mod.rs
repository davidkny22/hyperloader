//! Shared-memory regions, slots, and delivery-view lifetimes live in this module.

mod allocator;
mod named;
mod reaper;
mod writer;

pub use allocator::{
    ArenaAllocator, ArenaError, ArenaSizing, ArenaStats, DeliveryView, GrowthPolicy, SlabSpec,
    SlotRef, SlotState,
};
pub use named::{NamedRegion, RegionError, RegionName, RegionToken};
pub use reaper::{
    ProcessIdentity, ProcessObservation, ProcessObserver, RegionRegistry, RegistryEntry,
    RegistryError, RegistryIssue, RegistrySnapshot, SweepAction, SweepOutcome, SweepReport,
    SystemProcessObserver,
};
pub use writer::{ArenaWriter, ArenaWriterError};

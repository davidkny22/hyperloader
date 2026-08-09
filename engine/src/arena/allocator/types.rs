//! Public allocator vocabulary.

use crate::arena::RegistryError;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Whether an arena may add slabs after its initial plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GrowthPolicy {
    /// Add complete slabs when held views or variable-size values exhaust capacity.
    Safe,
    /// Return a typed error instead of allocating outside the initial plan.
    StrictError,
}

/// One initial slab class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SlabSpec {
    /// Payload bytes available to each slot.
    pub slot_capacity: usize,
    /// Fixed slot count in every slab grown from this class.
    pub slots_per_slab: u32,
    /// Whether this class is reserved for values beyond the fixed-size classes.
    pub overflow: bool,
}

/// Inputs and checked result for the plan-time arena byte formula.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArenaSizing {
    /// Maximum per-rank frontier depth in samples.
    pub depth_ceiling: usize,
    /// Per-rank batch size.
    pub batch_size: usize,
    /// Number of delivery-buffer batches.
    pub delivery_batches: usize,
    /// Planned bytes per sample.
    pub bytes_per_sample: usize,
}

impl ArenaSizing {
    /// Evaluate the complete planned arena footprint with checked arithmetic.
    pub fn arena_bytes(self) -> Result<usize, ArenaError> {
        let delivery = self
            .delivery_batches
            .checked_mul(self.batch_size)
            .ok_or(ArenaError::SizingOverflow)?;
        self.depth_ceiling
            .checked_add(delivery)
            .and_then(|slots| slots.checked_mul(self.bytes_per_sample))
            .ok_or(ArenaError::SizingOverflow)
    }
}

/// Stable coordinates for one slot in one immutable region.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SlotRef {
    /// Region sequence embedded in the portable name.
    pub region_sequence: u16,
    /// Complete payload size required for validated region attachment.
    pub region_size: u64,
    /// Zero-based slot within the slab.
    pub slot_index: u32,
    /// Byte offset within the region payload.
    pub offset: u64,
    /// Maximum byte length.
    pub capacity: u64,
    /// Reuse generation that rejects stale commands and views.
    pub generation: u64,
}

/// Observable ownership state for one slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlotState {
    /// Available for reservation.
    Free,
    /// Reserved by one worker identity.
    Writing { worker: u32 },
    /// Complete and waiting for delivery.
    Ready { length: usize },
    /// Delivered with this many tensor or structure references still live.
    Delivered { remaining: u32, length: usize },
    /// A worker died or abandoned the slot before completing it.
    Poisoned,
}

/// Current allocator counters used by telemetry and tests.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArenaStats {
    /// Complete slab regions currently owned by the allocator.
    pub regions: usize,
    /// Additional slab allocations after construction.
    pub growth_events: u64,
    /// Growth events caused by all matching slots being held or busy.
    pub hold_events: u64,
    /// Reservations routed beyond the fixed-size classes.
    pub overflow_events: u64,
    /// Slots currently poisoned and awaiting post-exception reclamation.
    pub poisoned_slots: usize,
}

/// A typed allocator, ownership, or growth failure.
#[derive(Debug)]
pub enum ArenaError {
    /// A size formula or slab layout overflowed `usize`.
    SizingOverflow,
    /// Slab classes are empty, malformed, unsorted, or lack one overflow class.
    InvalidSlabPlan,
    /// A reservation requested an empty payload.
    InvalidSize,
    /// Region creation or registry persistence failed.
    Registry(RegistryError),
    /// The two-character region sequence was exhausted.
    RegionLimit,
    /// The initial plan has no free slot and growth is forbidden.
    GrowthDenied { required: usize },
    /// A slot reference does not belong to this allocator or generation.
    StaleSlot,
    /// A transition was attempted from the wrong ownership state.
    InvalidTransition {
        expected: &'static str,
        actual: SlotState,
    },
    /// A different worker attempted to complete the reservation.
    WrongWriter { expected: u32, actual: u32 },
    /// The produced payload exceeds its reserved slot.
    SlotOverflow { capacity: usize, actual: usize },
    /// A synchronization primitive was poisoned by a panic.
    Synchronization,
}

impl Display for ArenaError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SizingOverflow => formatter.write_str("arena sizing overflowed"),
            Self::InvalidSlabPlan => formatter.write_str("arena slab plan is invalid"),
            Self::InvalidSize => formatter.write_str("arena reservation size must be positive"),
            Self::Registry(source) => {
                write!(formatter, "arena region registration failed: {source}")
            }
            Self::RegionLimit => formatter.write_str("arena region sequence is exhausted"),
            Self::GrowthDenied { required } => {
                write!(formatter, "arena growth is disabled for {required} bytes")
            }
            Self::StaleSlot => formatter.write_str("arena slot reference is stale"),
            Self::InvalidTransition { expected, actual } => {
                write!(formatter, "arena slot must be {expected}, not {actual:?}")
            }
            Self::WrongWriter { expected, actual } => write!(
                formatter,
                "arena slot belongs to worker {expected}, not worker {actual}"
            ),
            Self::SlotOverflow { capacity, actual } => write!(
                formatter,
                "arena slot capacity is {capacity} bytes but payload is {actual} bytes"
            ),
            Self::Synchronization => formatter.write_str("arena synchronization was poisoned"),
        }
    }
}

impl Error for ArenaError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Registry(source) => Some(source),
            _ => None,
        }
    }
}

impl From<RegistryError> for ArenaError {
    fn from(error: RegistryError) -> Self {
        Self::Registry(error)
    }
}

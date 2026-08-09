//! Size-classed arena slabs and explicit slot ownership transitions.

use super::{RegionRegistry, RegionToken, RegistryError};
use crate::arena::NamedRegion;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::num::NonZeroU32;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, MutexGuard};
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

const MAX_REGION_SEQUENCE: u32 = 1023;
#[derive(Clone, Copy, Debug)]
struct SlotMeta {
    generation: u64,
    state: SlotState,
}

struct Slab {
    spec: SlabSpec,
    sequence: u16,
    region: NamedRegion,
    registry: RegionRegistry,
    slots: Vec<SlotMeta>,
}

impl Slab {
    fn slot_ref(&self, index: usize) -> SlotRef {
        let slot = self.slots[index];
        SlotRef {
            region_sequence: self.sequence,
            slot_index: index as u32,
            offset: (index * self.spec.slot_capacity) as u64,
            capacity: self.spec.slot_capacity as u64,
            generation: slot.generation,
        }
    }
}

impl Drop for Slab {
    fn drop(&mut self) {
        let name = self.region.name().as_str().to_owned();
        let _ = self.region.unlink();
        let _ = self.registry.retain(|entry| entry.name != name);
    }
}

struct ArenaInner {
    registry: RegionRegistry,
    token: RegionToken,
    policy: GrowthPolicy,
    next_sequence: u32,
    max_standard_capacity: usize,
    slabs: Vec<Slab>,
    stats: ArenaStats,
}

impl ArenaInner {
    fn add_slab(&mut self, spec: SlabSpec, growth: bool) -> Result<usize, ArenaError> {
        if self.next_sequence > MAX_REGION_SEQUENCE {
            return Err(ArenaError::RegionLimit);
        }
        let payload_size = spec
            .slot_capacity
            .checked_mul(spec.slots_per_slab as usize)
            .ok_or(ArenaError::SizingOverflow)?;
        let sequence = self.next_sequence as u16;
        let (region, _) = self
            .registry
            .create_region(self.token, sequence, payload_size)?;
        self.next_sequence += 1;
        let slots = vec![
            SlotMeta {
                generation: 0,
                state: SlotState::Free,
            };
            spec.slots_per_slab as usize
        ];
        self.slabs.push(Slab {
            spec,
            sequence,
            region,
            registry: self.registry.clone(),
            slots,
        });
        if growth {
            self.stats.growth_events = self.stats.growth_events.saturating_add(1);
        }
        Ok(self.slabs.len() - 1)
    }
}

fn validate_specs(specs: &[SlabSpec]) -> Result<(), ArenaError> {
    if specs.is_empty()
        || specs
            .iter()
            .any(|spec| spec.slot_capacity == 0 || spec.slots_per_slab == 0)
        || specs
            .windows(2)
            .any(|pair| pair[0].slot_capacity >= pair[1].slot_capacity || pair[0].overflow)
        || specs.iter().filter(|spec| spec.overflow).count() != 1
        || !specs.last().is_some_and(|spec| spec.overflow)
    {
        return Err(ArenaError::InvalidSlabPlan);
    }
    Ok(())
}

fn find_free_slot(slabs: &[Slab], required: usize, overflow: bool) -> Option<(usize, usize)> {
    slabs
        .iter()
        .enumerate()
        .filter(|(_, slab)| slab.spec.overflow == overflow)
        .filter(|(_, slab)| slab.spec.slot_capacity >= required)
        .filter_map(|(slab_index, slab)| {
            slab.slots
                .iter()
                .position(|slot| slot.state == SlotState::Free)
                .map(|slot_index| (slab.spec.slot_capacity, slab_index, slot_index))
        })
        .min_by_key(|(capacity, _, _)| *capacity)
        .map(|(_, slab_index, slot_index)| (slab_index, slot_index))
}

fn reserve_slot(slab: &mut Slab, index: usize, worker: u32) -> Result<SlotRef, ArenaError> {
    let actual = slab.slots[index].state;
    if actual != SlotState::Free {
        return Err(ArenaError::InvalidTransition {
            expected: "free",
            actual,
        });
    }
    slab.slots[index].state = SlotState::Writing { worker };
    Ok(slab.slot_ref(index))
}

fn locate_slot(slabs: &[Slab], slot: SlotRef) -> Result<(usize, usize), ArenaError> {
    let slab_index = slabs
        .iter()
        .position(|slab| slab.sequence == slot.region_sequence)
        .ok_or(ArenaError::StaleSlot)?;
    let slot_index = slot.slot_index as usize;
    let slab = &slabs[slab_index];
    let meta = slab.slots.get(slot_index).ok_or(ArenaError::StaleSlot)?;
    if meta.generation != slot.generation
        || slot.offset != (slot_index * slab.spec.slot_capacity) as u64
        || slot.capacity != slab.spec.slot_capacity as u64
    {
        return Err(ArenaError::StaleSlot);
    }
    Ok((slab_index, slot_index))
}

fn recycle(meta: &mut SlotMeta) {
    meta.generation = meta.generation.wrapping_add(1);
    meta.state = SlotState::Free;
}

/// Thread-safe owner of size-classed, immutable shared-memory slab regions.
#[derive(Clone)]
pub struct ArenaAllocator {
    inner: Arc<Mutex<ArenaInner>>,
}

impl ArenaAllocator {
    /// Create one initial region per validated slab class.
    pub fn new(
        registry: RegionRegistry,
        token: RegionToken,
        specs: &[SlabSpec],
        policy: GrowthPolicy,
    ) -> Result<Self, ArenaError> {
        validate_specs(specs)?;
        let max_standard_capacity = specs
            .iter()
            .filter(|spec| !spec.overflow)
            .map(|spec| spec.slot_capacity)
            .max()
            .ok_or(ArenaError::InvalidSlabPlan)?;
        let mut inner = ArenaInner {
            registry,
            token,
            policy,
            next_sequence: 0,
            max_standard_capacity,
            slabs: Vec::new(),
            stats: ArenaStats::default(),
        };
        for spec in specs {
            inner.add_slab(*spec, false)?;
        }
        inner.stats.regions = inner.slabs.len();
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    /// Reserve the smallest fitting free slot for one worker.
    pub fn reserve(&self, required: usize, worker: u32) -> Result<SlotRef, ArenaError> {
        if required == 0 {
            return Err(ArenaError::InvalidSize);
        }
        let mut inner = self.lock()?;
        let overflow = required > inner.max_standard_capacity;
        if overflow {
            inner.stats.overflow_events = inner.stats.overflow_events.saturating_add(1);
        }
        if let Some((slab_index, slot_index)) = find_free_slot(&inner.slabs, required, overflow) {
            return reserve_slot(&mut inner.slabs[slab_index], slot_index, worker);
        }

        if inner.policy == GrowthPolicy::StrictError {
            return Err(ArenaError::GrowthDenied { required });
        }
        let matching = inner
            .slabs
            .iter()
            .filter(|slab| slab.spec.overflow == overflow)
            .filter(|slab| slab.spec.slot_capacity >= required)
            .min_by_key(|slab| slab.spec.slot_capacity)
            .map(|slab| slab.spec);
        let spec = match matching {
            Some(spec) => {
                let consumer_held = inner
                    .slabs
                    .iter()
                    .filter(|slab| slab.spec == spec)
                    .flat_map(|slab| slab.slots.iter())
                    .any(|slot| matches!(slot.state, SlotState::Delivered { .. }));
                if consumer_held {
                    inner.stats.hold_events = inner.stats.hold_events.saturating_add(1);
                }
                spec
            }
            None => SlabSpec {
                slot_capacity: required
                    .checked_next_power_of_two()
                    .ok_or(ArenaError::SizingOverflow)?,
                slots_per_slab: 1,
                overflow: true,
            },
        };
        let slab_index = inner.add_slab(spec, true)?;
        reserve_slot(&mut inner.slabs[slab_index], 0, worker)
    }

    /// Copy a completed payload and transition its reservation to ready.
    pub fn commit(&self, slot: SlotRef, worker: u32, payload: &[u8]) -> Result<(), ArenaError> {
        let mut inner = self.lock()?;
        let (slab_index, slot_index) = locate_slot(&inner.slabs, slot)?;
        let slab = &mut inner.slabs[slab_index];
        let meta = slab.slots[slot_index];
        let expected_worker = match meta.state {
            SlotState::Writing { worker } => worker,
            actual => {
                return Err(ArenaError::InvalidTransition {
                    expected: "writing",
                    actual,
                });
            }
        };
        if expected_worker != worker {
            return Err(ArenaError::WrongWriter {
                expected: expected_worker,
                actual: worker,
            });
        }
        if payload.len() > slab.spec.slot_capacity {
            return Err(ArenaError::SlotOverflow {
                capacity: slab.spec.slot_capacity,
                actual: payload.len(),
            });
        }
        let start = slot.offset as usize;
        let end = start + payload.len();
        // SAFETY: the slot is in the exclusive writing state for this worker and the range
        // was checked against its immutable capacity.
        unsafe { slab.region.payload_mut()[start..end].copy_from_slice(payload) };
        slab.slots[slot_index].state = SlotState::Ready {
            length: payload.len(),
        };
        Ok(())
    }

    /// Deliver one ready slot as independently refcounted views.
    pub fn deliver(
        &self,
        slot: SlotRef,
        references: NonZeroU32,
    ) -> Result<Vec<DeliveryView>, ArenaError> {
        let mut inner = self.lock()?;
        let (slab_index, slot_index) = locate_slot(&inner.slabs, slot)?;
        let length = match inner.slabs[slab_index].slots[slot_index].state {
            SlotState::Ready { length } => length,
            actual => {
                return Err(ArenaError::InvalidTransition {
                    expected: "ready",
                    actual,
                });
            }
        };
        inner.slabs[slab_index].slots[slot_index].state = SlotState::Delivered {
            remaining: references.get(),
            length,
        };
        drop(inner);
        Ok((0..references.get())
            .map(|_| DeliveryView {
                inner: Arc::clone(&self.inner),
                slot,
                released: false,
            })
            .collect())
    }

    /// Poison every incomplete slot owned by a failed worker.
    pub fn poison_writer(&self, worker: u32) -> Result<Vec<SlotRef>, ArenaError> {
        let mut inner = self.lock()?;
        let mut poisoned = Vec::new();
        for slab in &mut inner.slabs {
            for index in 0..slab.slots.len() {
                if slab.slots[index].state == (SlotState::Writing { worker }) {
                    slab.slots[index].state = SlotState::Poisoned;
                    poisoned.push(slab.slot_ref(index));
                }
            }
        }
        inner.stats.poisoned_slots = inner.stats.poisoned_slots.saturating_add(poisoned.len());
        Ok(poisoned)
    }

    /// Reclaim one poisoned slot after its exception has surfaced.
    pub fn reclaim_poisoned(&self, slot: SlotRef) -> Result<(), ArenaError> {
        let mut inner = self.lock()?;
        let (slab_index, slot_index) = locate_slot(&inner.slabs, slot)?;
        let meta = &mut inner.slabs[slab_index].slots[slot_index];
        if meta.state != SlotState::Poisoned {
            return Err(ArenaError::InvalidTransition {
                expected: "poisoned",
                actual: meta.state,
            });
        }
        recycle(meta);
        inner.stats.poisoned_slots = inner.stats.poisoned_slots.saturating_sub(1);
        Ok(())
    }

    /// Return allocator counters and current region count.
    pub fn stats(&self) -> Result<ArenaStats, ArenaError> {
        let inner = self.lock()?;
        let mut stats = inner.stats;
        stats.regions = inner.slabs.len();
        Ok(stats)
    }

    /// Inspect one current slot state for diagnostics.
    pub fn slot_state(&self, slot: SlotRef) -> Result<SlotState, ArenaError> {
        let inner = self.lock()?;
        let (slab_index, slot_index) = locate_slot(&inner.slabs, slot)?;
        Ok(inner.slabs[slab_index].slots[slot_index].state)
    }

    fn lock(&self) -> Result<MutexGuard<'_, ArenaInner>, ArenaError> {
        self.inner.lock().map_err(|_| ArenaError::Synchronization)
    }
}

/// One consumer-held reference whose final drop recycles the slot.
pub struct DeliveryView {
    inner: Arc<Mutex<ArenaInner>>,
    slot: SlotRef,
    released: bool,
}

impl DeliveryView {
    /// Return the stable slot coordinates backing this view.
    pub const fn slot(&self) -> SlotRef {
        self.slot
    }

    /// Copy the immutable delivered bytes for validation or metadata paths.
    pub fn to_vec(&self) -> Result<Vec<u8>, ArenaError> {
        let inner = self.inner.lock().map_err(|_| ArenaError::Synchronization)?;
        let (slab_index, slot_index) = locate_slot(&inner.slabs, self.slot)?;
        let length = match inner.slabs[slab_index].slots[slot_index].state {
            SlotState::Delivered { length, .. } => length,
            actual => {
                return Err(ArenaError::InvalidTransition {
                    expected: "delivered",
                    actual,
                });
            }
        };
        let start = self.slot.offset as usize;
        let end = start + length;
        // SAFETY: delivered slots are immutable until every view drops, and this view holds one
        // reference while the copy is made under the metadata lock.
        Ok(unsafe { inner.slabs[slab_index].region.payload()[start..end].to_vec() })
    }

    /// Return a stable zero-copy pointer and length for a foreign storage wrapper.
    ///
    /// # Safety
    ///
    /// The caller must not outlive this view and must not mutate the returned bytes. The foreign
    /// storage deleter must own and drop this `DeliveryView` exactly once.
    pub unsafe fn as_ptr_len(&self) -> Result<(NonNull<u8>, usize), ArenaError> {
        let inner = self.inner.lock().map_err(|_| ArenaError::Synchronization)?;
        let (slab_index, slot_index) = locate_slot(&inner.slabs, self.slot)?;
        let length = match inner.slabs[slab_index].slots[slot_index].state {
            SlotState::Delivered { length, .. } => length,
            actual => {
                return Err(ArenaError::InvalidTransition {
                    expected: "delivered",
                    actual,
                });
            }
        };
        // SAFETY: the mapping remains alive through `inner`, the slot is delivered and immutable,
        // and the checked slot offset lies within its region payload.
        let pointer = unsafe {
            inner.slabs[slab_index]
                .region
                .payload()
                .as_ptr()
                .add(self.slot.offset as usize)
                .cast_mut()
        };
        Ok((
            NonNull::new(pointer).expect("mapped payload pointer is non-null"),
            length,
        ))
    }

    fn release(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        let Ok((slab_index, slot_index)) = locate_slot(&inner.slabs, self.slot) else {
            return;
        };
        let meta = &mut inner.slabs[slab_index].slots[slot_index];
        let SlotState::Delivered { remaining, length } = meta.state else {
            return;
        };
        if remaining == 1 {
            recycle(meta);
        } else {
            meta.state = SlotState::Delivered {
                remaining: remaining - 1,
                length,
            };
        }
    }
}

impl Drop for DeliveryView {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::{ArenaAllocator, ArenaError, ArenaSizing, GrowthPolicy, SlabSpec, SlotState};
    use crate::arena::{RegionRegistry, RegionToken};
    use std::num::NonZeroU32;

    fn allocator(policy: GrowthPolicy) -> ArenaAllocator {
        let mut random = [0_u8; 8];
        getrandom::fill(&mut random).expect("registry suffix");
        let suffix: String = random.iter().map(|byte| format!("{byte:02x}")).collect();
        let registry = RegionRegistry::new(
            std::env::temp_dir().join(format!("hl-slabs-{suffix}/regions.jsonl")),
        );
        ArenaAllocator::new(
            registry,
            RegionToken::random().expect("arena token"),
            &[
                SlabSpec {
                    slot_capacity: 16,
                    slots_per_slab: 2,
                    overflow: false,
                },
                SlabSpec {
                    slot_capacity: 64,
                    slots_per_slab: 1,
                    overflow: true,
                },
            ],
            policy,
        )
        .expect("arena allocator")
    }

    #[test]
    fn sizing_formula_is_checked() {
        assert_eq!(
            ArenaSizing {
                depth_ceiling: 8,
                batch_size: 4,
                delivery_batches: 2,
                bytes_per_sample: 16,
            }
            .arena_bytes()
            .expect("sizing"),
            256
        );
        assert!(matches!(
            ArenaSizing {
                depth_ceiling: usize::MAX,
                batch_size: 1,
                delivery_batches: 1,
                bytes_per_sample: 1,
            }
            .arena_bytes(),
            Err(ArenaError::SizingOverflow)
        ));
    }

    #[test]
    fn empty_reservations_are_rejected() {
        let arena = allocator(GrowthPolicy::Safe);
        assert!(matches!(arena.reserve(0, 1), Err(ArenaError::InvalidSize)));
    }

    #[test]
    fn delivery_views_recycle_only_after_final_release() {
        let arena = allocator(GrowthPolicy::Safe);
        let slot = arena.reserve(4, 7).expect("reserve slot");
        arena.commit(slot, 7, b"data").expect("commit payload");
        let mut views = arena
            .deliver(slot, NonZeroU32::new(2).expect("nonzero references"))
            .expect("deliver views");
        assert_eq!(views[0].to_vec().expect("first view"), b"data");
        drop(views.pop());
        assert_eq!(
            arena.slot_state(slot).expect("held state"),
            SlotState::Delivered {
                remaining: 1,
                length: 4,
            }
        );
        drop(views);
        assert!(matches!(arena.slot_state(slot), Err(ArenaError::StaleSlot)));

        let reused = arena.reserve(4, 8).expect("reuse slot");
        assert_eq!(reused.slot_index, slot.slot_index);
        assert_eq!(reused.generation, slot.generation + 1);
    }

    #[test]
    fn wrong_writer_and_overrun_do_not_publish() {
        let arena = allocator(GrowthPolicy::Safe);
        let slot = arena.reserve(8, 9).expect("reserve slot");
        assert!(matches!(
            arena.commit(slot, 10, b"data"),
            Err(ArenaError::WrongWriter { .. })
        ));
        assert!(matches!(
            arena.commit(slot, 9, &[0; 17]),
            Err(ArenaError::SlotOverflow { .. })
        ));
        assert_eq!(
            arena.slot_state(slot).expect("writing state"),
            SlotState::Writing { worker: 9 }
        );
    }

    #[test]
    fn worker_poison_requires_explicit_post_exception_reclaim() {
        let arena = allocator(GrowthPolicy::Safe);
        let first = arena.reserve(8, 11).expect("first worker slot");
        let second = arena.reserve(8, 11).expect("second worker slot");
        let poisoned = arena.poison_writer(11).expect("poison worker slots");
        assert_eq!(poisoned, [first, second]);
        assert_eq!(arena.stats().expect("stats").poisoned_slots, 2);
        assert!(matches!(
            arena.commit(first, 11, b"late"),
            Err(ArenaError::InvalidTransition { .. })
        ));
        arena
            .reclaim_poisoned(first)
            .expect("reclaim after exception");
        assert_eq!(arena.stats().expect("stats").poisoned_slots, 1);
        assert!(matches!(
            arena.slot_state(first),
            Err(ArenaError::StaleSlot)
        ));
    }

    #[test]
    fn held_slots_grow_by_complete_slab_under_safe_policy() {
        let arena = allocator(GrowthPolicy::Safe);
        let first = arena.reserve(8, 1).expect("first slot");
        let second = arena.reserve(8, 2).expect("second slot");
        arena.commit(first, 1, b"first").expect("first commit");
        arena.commit(second, 2, b"second").expect("second commit");
        let first_view = arena
            .deliver(first, NonZeroU32::new(1).expect("reference"))
            .expect("first delivery");
        let second_view = arena
            .deliver(second, NonZeroU32::new(1).expect("reference"))
            .expect("second delivery");
        let third = arena.reserve(8, 3).expect("growth slot");
        assert_ne!(third.region_sequence, first.region_sequence);
        assert_ne!(third.region_sequence, second.region_sequence);
        let stats = arena.stats().expect("growth stats");
        assert_eq!(stats.growth_events, 1);
        assert_eq!(stats.hold_events, 1);
        assert_eq!(stats.regions, 3);
        drop((first_view, second_view));
    }

    #[test]
    fn busy_writers_grow_without_reporting_a_consumer_hold() {
        let arena = allocator(GrowthPolicy::Safe);
        arena.reserve(8, 1).expect("first slot");
        arena.reserve(8, 2).expect("second slot");
        arena.reserve(8, 3).expect("growth slot");
        let stats = arena.stats().expect("growth stats");
        assert_eq!(stats.growth_events, 1);
        assert_eq!(stats.hold_events, 0);
    }

    #[test]
    fn strict_policy_rejects_hold_and_variable_growth() {
        let arena = allocator(GrowthPolicy::StrictError);
        let first = arena.reserve(8, 1).expect("first slot");
        let second = arena.reserve(8, 2).expect("second slot");
        arena.commit(first, 1, b"first").expect("first commit");
        arena.commit(second, 2, b"second").expect("second commit");
        let first_view = arena
            .deliver(first, NonZeroU32::new(1).expect("reference"))
            .expect("first delivery");
        let second_view = arena
            .deliver(second, NonZeroU32::new(1).expect("reference"))
            .expect("second delivery");
        assert!(matches!(
            arena.reserve(8, 3),
            Err(ArenaError::GrowthDenied { required: 8 })
        ));
        assert!(matches!(
            arena.reserve(65, 4),
            Err(ArenaError::GrowthDenied { required: 65 })
        ));
        assert_eq!(arena.stats().expect("strict stats").growth_events, 0);
        drop((first_view, second_view));
    }

    #[test]
    fn overflow_reservations_are_counted_and_grow_when_needed() {
        let arena = allocator(GrowthPolicy::Safe);
        let first = arena.reserve(32, 1).expect("initial overflow slot");
        assert_eq!(first.capacity, 64);
        let second = arena.reserve(80, 2).expect("grown overflow slot");
        assert_eq!(second.capacity, 128);
        let stats = arena.stats().expect("overflow stats");
        assert_eq!(stats.overflow_events, 2);
        assert_eq!(stats.growth_events, 1);
        assert_eq!(stats.hold_events, 0);
    }

    #[test]
    fn final_region_sequence_is_usable_without_partial_creation() {
        let arena = allocator(GrowthPolicy::Safe);
        let mut inner = arena.lock().expect("arena lock");
        inner.next_sequence = super::MAX_REGION_SEQUENCE;
        let spec = SlabSpec {
            slot_capacity: 128,
            slots_per_slab: 1,
            overflow: true,
        };
        let final_index = inner.add_slab(spec, true).expect("final region");
        assert_eq!(inner.slabs[final_index].sequence, 1023);
        assert!(matches!(
            inner.add_slab(spec, true),
            Err(ArenaError::RegionLimit)
        ));
        assert_eq!(
            inner.registry.snapshot().expect("registry").entries.len(),
            3
        );
    }
}

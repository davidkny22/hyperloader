//! Size-classed arena slabs and explicit slot ownership transitions.

mod delivery;
mod slab;
mod types;

pub use delivery::DeliveryView;
pub use types::{ArenaError, ArenaSizing, ArenaStats, GrowthPolicy, SlabSpec, SlotRef, SlotState};

use crate::arena::{RegionRegistry, RegionToken};
#[cfg(test)]
use slab::MAX_REGION_SEQUENCE;
use slab::{ArenaInner, find_free_slot, locate_slot, recycle, reserve_slot, validate_specs};
use std::num::NonZeroU32;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, MutexGuard};

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
        Self::new_at_sequence(registry, token, specs, policy, 0)
    }

    /// Create initial slabs beginning at a sequence reserved by the loader's region plan.
    pub fn new_at_sequence(
        registry: RegionRegistry,
        token: RegionToken,
        specs: &[SlabSpec],
        policy: GrowthPolicy,
        first_sequence: u16,
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
            next_sequence: u32::from(first_sequence),
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

    /// Return the stable writable range of one exclusive reservation.
    ///
    /// # Safety
    ///
    /// The caller must keep this allocator alive, publish or cancel the reservation exactly
    /// once, and never write after publication. The slot remains exclusively owned by `worker`
    /// until that transition.
    pub unsafe fn writing_ptr_len(
        &self,
        slot: SlotRef,
        worker: u32,
    ) -> Result<(NonNull<u8>, usize), ArenaError> {
        let mut inner = self.lock()?;
        let (slab_index, slot_index) = locate_slot(&inner.slabs, slot)?;
        let slab = &mut inner.slabs[slab_index];
        validate_writer(slab.slots[slot_index].state, worker)?;
        let pointer = unsafe {
            slab.region
                .payload_mut()
                .as_mut_ptr()
                .add(slot.offset as usize)
        };
        Ok((
            NonNull::new(pointer).expect("mapped payload pointer is non-null"),
            slot.capacity as usize,
        ))
    }

    /// Copy a completed payload and transition its reservation to ready.
    pub fn commit(&self, slot: SlotRef, worker: u32, payload: &[u8]) -> Result<(), ArenaError> {
        let mut inner = self.lock()?;
        let (slab_index, slot_index) = locate_slot(&inner.slabs, slot)?;
        let slab = &mut inner.slabs[slab_index];
        validate_writer(slab.slots[slot_index].state, worker)?;
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

    /// Publish bytes written directly by an attached worker into its reserved slot.
    pub fn publish(&self, slot: SlotRef, worker: u32, length: usize) -> Result<(), ArenaError> {
        let mut inner = self.lock()?;
        let (slab_index, slot_index) = locate_slot(&inner.slabs, slot)?;
        let slab = &mut inner.slabs[slab_index];
        validate_writer(slab.slots[slot_index].state, worker)?;
        if length > slab.spec.slot_capacity {
            return Err(ArenaError::SlotOverflow {
                capacity: slab.spec.slot_capacity,
                actual: length,
            });
        }
        slab.slots[slot_index].state = SlotState::Ready { length };
        Ok(())
    }

    /// Recycle a reservation that a worker intentionally left unused.
    pub fn cancel(&self, slot: SlotRef, worker: u32) -> Result<(), ArenaError> {
        let mut inner = self.lock()?;
        let (slab_index, slot_index) = locate_slot(&inner.slabs, slot)?;
        let meta = &mut inner.slabs[slab_index].slots[slot_index];
        validate_writer(meta.state, worker)?;
        recycle(meta);
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

fn validate_writer(state: SlotState, worker: u32) -> Result<(), ArenaError> {
    let expected_worker = match state {
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
    Ok(())
}

#[cfg(test)]
mod tests;

//! Physical slab ownership and slot bookkeeping.

use super::{ArenaError, ArenaStats, GrowthPolicy, SlabSpec, SlotRef, SlotState};
use crate::arena::{NamedRegion, RegionRegistry, RegionToken};

pub(super) const MAX_REGION_SEQUENCE: u32 = 1023;

#[derive(Clone, Copy, Debug)]
pub(super) struct SlotMeta {
    pub(super) generation: u64,
    pub(super) state: SlotState,
}

pub(super) struct Slab {
    pub(super) spec: SlabSpec,
    pub(super) sequence: u16,
    pub(super) region: NamedRegion,
    pub(super) registry: RegionRegistry,
    pub(super) slots: Vec<SlotMeta>,
}

impl Slab {
    pub(super) fn slot_ref(&self, index: usize) -> SlotRef {
        let slot = self.slots[index];
        SlotRef {
            region_sequence: self.sequence,
            region_size: self.region.payload_size() as u64,
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

pub(super) struct ArenaInner {
    pub(super) registry: RegionRegistry,
    pub(super) token: RegionToken,
    pub(super) policy: GrowthPolicy,
    pub(super) next_sequence: u32,
    pub(super) max_standard_capacity: usize,
    pub(super) slabs: Vec<Slab>,
    pub(super) stats: ArenaStats,
}

impl ArenaInner {
    pub(super) fn add_slab(&mut self, spec: SlabSpec, growth: bool) -> Result<usize, ArenaError> {
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

pub(super) fn validate_specs(specs: &[SlabSpec]) -> Result<(), ArenaError> {
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

pub(super) fn find_free_slot(
    slabs: &[Slab],
    required: usize,
    overflow: bool,
) -> Option<(usize, usize)> {
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

pub(super) fn reserve_slot(
    slab: &mut Slab,
    index: usize,
    worker: u32,
) -> Result<SlotRef, ArenaError> {
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

pub(super) fn locate_slot(slabs: &[Slab], slot: SlotRef) -> Result<(usize, usize), ArenaError> {
    let slab_index = slabs
        .iter()
        .position(|slab| slab.sequence == slot.region_sequence)
        .ok_or(ArenaError::StaleSlot)?;
    let slot_index = slot.slot_index as usize;
    let slab = &slabs[slab_index];
    let meta = slab.slots.get(slot_index).ok_or(ArenaError::StaleSlot)?;
    if meta.generation != slot.generation
        || slot.region_size != slab.region.payload_size() as u64
        || slot.offset != (slot_index * slab.spec.slot_capacity) as u64
        || slot.capacity != slab.spec.slot_capacity as u64
    {
        return Err(ArenaError::StaleSlot);
    }
    Ok((slab_index, slot_index))
}

pub(super) fn recycle(meta: &mut SlotMeta) {
    meta.generation = meta.generation.wrapping_add(1);
    meta.state = SlotState::Free;
}

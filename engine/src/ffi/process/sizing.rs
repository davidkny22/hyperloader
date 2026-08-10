//! Process transport slab sizing by reservation concern.

use crate::arena::{ArenaError, SlabSpec};

pub(super) fn process_slab_specs(
    worker_count: u32,
    queue_capacity: usize,
    payload_capacity: usize,
    exception_capacity: usize,
) -> Result<Vec<SlabSpec>, ArenaError> {
    let queue_slots = u32::try_from(queue_capacity).map_err(|_| ArenaError::SizingOverflow)?;
    let slots_per_class = queue_slots;
    let largest_capacity = payload_capacity.max(exception_capacity);
    let overflow_capacity = largest_capacity
        .checked_mul(2)
        .ok_or(ArenaError::SizingOverflow)?;

    let mut specs = if payload_capacity == exception_capacity {
        vec![SlabSpec {
            slot_capacity: payload_capacity,
            slots_per_slab: slots_per_class
                .checked_mul(2)
                .ok_or(ArenaError::SizingOverflow)?,
            overflow: false,
        }]
    } else {
        let mut fixed = [
            SlabSpec {
                slot_capacity: payload_capacity,
                slots_per_slab: slots_per_class,
                overflow: false,
            },
            SlabSpec {
                slot_capacity: exception_capacity,
                slots_per_slab: slots_per_class,
                overflow: false,
            },
        ];
        fixed.sort_by_key(|spec| spec.slot_capacity);
        fixed.to_vec()
    };
    specs.push(SlabSpec {
        slot_capacity: overflow_capacity,
        slots_per_slab: worker_count,
        overflow: true,
    });
    Ok(specs)
}

#[cfg(test)]
mod tests;

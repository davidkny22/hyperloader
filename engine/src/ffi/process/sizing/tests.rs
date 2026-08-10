use super::process_slab_specs;
use crate::arena::{ArenaError, SlabSpec};

#[test]
fn bounds_each_primary_and_exception_slab_to_one_worker_queue() {
    let specs = process_slab_specs(4, 128, 262_144, 65_536).unwrap();

    assert_eq!(
        specs,
        vec![
            SlabSpec {
                slot_capacity: 65_536,
                slots_per_slab: 128,
                overflow: false,
            },
            SlabSpec {
                slot_capacity: 262_144,
                slots_per_slab: 128,
                overflow: false,
            },
            SlabSpec {
                slot_capacity: 524_288,
                slots_per_slab: 4,
                overflow: true,
            },
        ]
    );
}

#[test]
fn merges_equal_fixed_capacities_without_losing_slots() {
    let specs = process_slab_specs(2, 8, 1_024, 1_024).unwrap();

    assert_eq!(specs[0].slots_per_slab, 16);
    assert_eq!(specs[0].slot_capacity, 1_024);
    assert_eq!(specs[1].slot_capacity, 2_048);
    assert!(specs[1].overflow);
}

#[test]
fn rejects_unrepresentable_queue_capacity() {
    let error = process_slab_specs(2, usize::MAX, 1, 2).unwrap_err();

    assert!(matches!(error, ArenaError::SizingOverflow));
}

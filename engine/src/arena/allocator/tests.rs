use super::{ArenaAllocator, ArenaError, GrowthPolicy, SlabSpec};
use crate::arena::{RegionRegistry, RegionToken};

fn allocator(policy: GrowthPolicy) -> ArenaAllocator {
    let mut random = [0_u8; 8];
    getrandom::fill(&mut random).expect("registry suffix");
    let suffix: String = random.iter().map(|byte| format!("{byte:02x}")).collect();
    let registry =
        RegionRegistry::new(std::env::temp_dir().join(format!("hl-slabs-{suffix}/regions.jsonl")));
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

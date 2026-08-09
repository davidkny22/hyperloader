use _hyperloader::arena::{
    ArenaAllocator, ArenaError, ArenaSizing, GrowthPolicy, RegionRegistry, RegionToken, SlabSpec,
    SlotState,
};
use std::num::NonZeroU32;

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

use _hyperloader::arena::{
    ArenaAllocator, ArenaError, ArenaWriter, ArenaWriterError, GrowthPolicy, RegionRegistry,
    RegionToken, SlabSpec, SlotState,
};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn attached_writer_publishes_reserved_bytes_without_owner_copy() {
    let token = RegionToken::random().expect("arena token");
    let arena = ArenaAllocator::new_at_sequence(
        registry("publish"),
        token,
        &[
            SlabSpec {
                slot_capacity: 64,
                slots_per_slab: 4,
                overflow: false,
            },
            SlabSpec {
                slot_capacity: 128,
                slots_per_slab: 2,
                overflow: true,
            },
        ],
        GrowthPolicy::Safe,
        5,
    )
    .expect("arena");
    let first = arena.reserve(32, 7).expect("first slot");
    assert_eq!(first.region_sequence, 5);
    assert_eq!(first.region_size, 256);
    let second = arena.reserve(32, 7).expect("second slot");
    let mut writer = ArenaWriter::new(token);
    writer.write(first, b"first payload").expect("first write");
    writer.write_at(second, 0, b"second ").expect("second row");
    writer.write_at(second, 7, b"payload").expect("third row");
    arena
        .publish(first, 7, b"first payload".len())
        .expect("publish first");
    arena
        .publish(second, 7, b"second payload".len())
        .expect("publish second");
    let first_view = arena
        .deliver(first, NonZeroU32::new(1).expect("one view"))
        .expect("deliver first")
        .pop()
        .expect("first view");
    let second_view = arena
        .deliver(second, NonZeroU32::new(1).expect("one view"))
        .expect("deliver second")
        .pop()
        .expect("second view");
    assert_eq!(first_view.to_vec().expect("first bytes"), b"first payload");
    assert_eq!(
        second_view.to_vec().expect("second bytes"),
        b"second payload"
    );
}

#[test]
fn attached_worker_reads_owner_command_before_overwriting_the_slot() {
    let token = RegionToken::random().expect("arena token");
    let arena = ArenaAllocator::new(
        registry("command-read"),
        token,
        &[
            SlabSpec {
                slot_capacity: 64,
                slots_per_slab: 1,
                overflow: false,
            },
            SlabSpec {
                slot_capacity: 128,
                slots_per_slab: 1,
                overflow: true,
            },
        ],
        GrowthPolicy::StrictError,
    )
    .expect("arena");
    let slot = arena.reserve(64, 3).expect("command slot");
    let command = b"sampler batch indices";
    // SAFETY: this fresh reservation remains owner-exclusive until the copy finishes.
    let (pointer, capacity) =
        unsafe { arena.writing_ptr_len(slot, 3).expect("command write range") };
    assert!(command.len() <= capacity);
    // SAFETY: the command fits the validated exclusive reservation.
    unsafe {
        std::ptr::copy_nonoverlapping(command.as_ptr(), pointer.as_ptr(), command.len());
    }
    let mut worker = ArenaWriter::new(token);
    assert_eq!(
        worker.read(slot, command.len()).expect("command read"),
        command
    );
    worker.write(slot, b"result").expect("result overwrite");
    arena.publish(slot, 3, 6).expect("publish result");
    let view = arena
        .deliver(slot, NonZeroU32::new(1).expect("one view"))
        .expect("deliver result")
        .pop()
        .expect("result view");
    assert_eq!(view.to_vec().expect("result bytes"), b"result");
}

#[test]
fn cancel_recycles_an_unused_worker_reservation() {
    let token = RegionToken::random().expect("arena token");
    let arena = ArenaAllocator::new(
        registry("cancel"),
        token,
        &[
            SlabSpec {
                slot_capacity: 64,
                slots_per_slab: 1,
                overflow: false,
            },
            SlabSpec {
                slot_capacity: 128,
                slots_per_slab: 1,
                overflow: true,
            },
        ],
        GrowthPolicy::StrictError,
    )
    .expect("arena");
    let slot = arena.reserve(16, 2).expect("reservation");
    assert!(matches!(
        arena.publish(slot, 3, 8),
        Err(ArenaError::WrongWriter {
            expected: 2,
            actual: 3
        })
    ));
    assert!(matches!(
        arena.publish(slot, 2, 65),
        Err(ArenaError::SlotOverflow {
            capacity: 64,
            actual: 65
        })
    ));
    arena.cancel(slot, 2).expect("cancel reservation");
    assert!(
        arena.slot_state(slot).is_err(),
        "the canceled generation must be stale"
    );
    let reused = arena.reserve(16, 2).expect("reused reservation");
    assert_eq!(reused.generation, slot.generation + 1);
    assert_eq!(
        arena.slot_state(reused).expect("reused state"),
        SlotState::Writing { worker: 2 }
    );
}

#[test]
fn writer_rejects_forged_region_bounds_before_attachment() {
    let mut writer = ArenaWriter::new(RegionToken::from_bytes([0x44; 16]));
    let slot = _hyperloader::arena::SlotRef {
        region_sequence: 1,
        region_size: 32,
        slot_index: 0,
        offset: 16,
        capacity: 32,
        generation: 0,
    };
    assert!(matches!(
        writer.write(slot, b"payload"),
        Err(ArenaWriterError::InvalidSlot("slot range"))
    ));
}

fn registry(label: &str) -> RegionRegistry {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    RegionRegistry::new(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("arena-writer-tests")
            .join(format!("{label}-{}-{unique}", std::process::id()))
            .join("regions.jsonl"),
    )
}

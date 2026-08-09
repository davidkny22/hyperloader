use _hyperloader::arena::{RegionRegistry, RegionToken, SlotRef};
use _hyperloader::exec::{CommandTransport, CompletionMessage, CompletionStatus, DispatchMessage};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) fn transport(
    label: &str,
    sequence: u16,
    dispatch_capacity: usize,
    completion_capacity: usize,
) -> CommandTransport {
    CommandTransport::create(
        registry(label),
        RegionToken::random().expect("transport token"),
        sequence,
        dispatch_capacity,
        completion_capacity,
    )
    .expect("create transport")
}

pub(super) fn registry(label: &str) -> RegionRegistry {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("command-transport-tests")
        .join(format!("{label}-{}-{unique}", std::process::id()))
        .join("regions.jsonl");
    RegionRegistry::new(path)
}

pub(super) fn dispatch(position: u64, worker: u32) -> DispatchMessage {
    DispatchMessage {
        position,
        stage_plan: 5,
        worker,
        slot: slot(position),
    }
}

pub(super) fn ready_completion(position: u64, worker: u32) -> CompletionMessage {
    CompletionMessage {
        position,
        worker,
        status: CompletionStatus::Ready,
        slot: slot(position),
        produced_length: 8,
        exception: None,
    }
}

pub(super) fn slot(seed: u64) -> SlotRef {
    SlotRef {
        region_sequence: (seed % 1024) as u16,
        slot_index: (seed % 31) as u32,
        offset: seed.saturating_mul(64),
        capacity: 64,
        generation: seed / 31,
    }
}

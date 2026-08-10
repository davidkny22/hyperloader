use super::common::registry;
use _hyperloader::arena::{ArenaAllocator, GrowthPolicy, RegionToken, SlabSpec};
use _hyperloader::exec::{CommandTransport, CompletionMessage, CompletionStatus, ExceptionRef};
use std::num::NonZeroU32;

#[test]
fn completion_carries_exception_bytes_by_side_slab_reference() {
    let registry = registry("side-slab");
    let token = RegionToken::random().expect("transport token");
    let arena = ArenaAllocator::new(
        registry.clone(),
        token,
        &[
            SlabSpec {
                slot_capacity: 64,
                slots_per_slab: 4,
                overflow: false,
            },
            SlabSpec {
                slot_capacity: 256,
                slots_per_slab: 2,
                overflow: true,
            },
        ],
        GrowthPolicy::Safe,
    )
    .expect("arena");
    let transport = CommandTransport::create(registry, token, 10, 4, 4).expect("transport");
    let primary = arena.reserve(32, 7).expect("primary slot");
    let exception_slot = arena.reserve(32, 7).expect("exception slot");
    let traceback = b"ValueError: invalid sample";
    arena
        .commit(exception_slot, 7, traceback)
        .expect("commit exception bytes");
    let message = CompletionMessage {
        position: 44,
        worker: 7,
        cost_ns: 900,
        status: CompletionStatus::Exception,
        slot: primary,
        produced_length: 0,
        exception: Some(ExceptionRef {
            slot: exception_slot,
            length: traceback.len() as u64,
        }),
    };
    transport
        .try_send_completion(message)
        .expect("send exception completion");
    let received = transport
        .try_recv_completion()
        .expect("receive exception completion");
    assert_eq!(received, message);
    let side = received.exception.expect("side-slab reference");
    let view = arena
        .deliver(side.slot, NonZeroU32::new(1).expect("one reference"))
        .expect("deliver side slab")
        .pop()
        .expect("exception view");
    assert_eq!(view.to_vec().expect("read exception bytes"), traceback);
}

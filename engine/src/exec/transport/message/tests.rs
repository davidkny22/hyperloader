use super::{decode_completion, decode_dispatch, encode_completion, encode_dispatch};
use crate::arena::SlotRef;
use crate::exec::{CompletionMessage, CompletionStatus, DispatchMessage};

#[test]
fn dispatch_decoder_rejects_corrupted_header_and_reserved_bytes() {
    let message = DispatchMessage {
        position: 8,
        epoch: 4,
        stage_plan: 3,
        index: 19,
        worker: 2,
        batch_len: 64,
        slot: slot(),
        exception_slot: SlotRef {
            slot_index: 3,
            ..slot()
        },
    };
    let frame = encode_dispatch(message).expect("dispatch frame");
    assert_eq!(decode_dispatch(&frame), Ok(message));
    let mut bad_magic = encode_dispatch(message).expect("dispatch frame");
    bad_magic[0] ^= 1;
    assert_eq!(decode_dispatch(&bad_magic), Err("magic"));

    let mut bad_reserved = encode_dispatch(message).expect("dispatch frame");
    bad_reserved[127] = 1;
    assert_eq!(
        decode_dispatch(&bad_reserved),
        Err("dispatch reserved fields")
    );
}

#[test]
fn completion_decoder_rejects_status_and_length_corruption() {
    let message = CompletionMessage {
        position: 8,
        worker: 2,
        cost_ns: 400,
        status: CompletionStatus::Ready,
        slot: slot(),
        produced_length: 16,
        exception: None,
    };
    let mut bad_status = encode_completion(message).expect("completion frame");
    bad_status[6] = 99;
    assert_eq!(decode_completion(&bad_status), Err("completion status"));

    let mut bad_length = encode_completion(message).expect("completion frame");
    bad_length[72..80].copy_from_slice(&65_u64.to_le_bytes());
    assert_eq!(decode_completion(&bad_length), Err("produced length"));

    let mut missing_cost = message;
    missing_cost.cost_ns = 0;
    assert!(matches!(
        encode_completion(missing_cost),
        Err(crate::exec::TransportError::InvalidMessage(
            "completion cost"
        ))
    ));
}

#[test]
fn raw_batch_completion_round_trips_its_distinct_status() {
    let message = CompletionMessage {
        position: 5,
        worker: 1,
        cost_ns: 700,
        status: CompletionStatus::ReadyBatch,
        slot: slot(),
        produced_length: 64,
        exception: None,
    };

    let frame = encode_completion(message).expect("batch completion frame");

    assert_eq!(decode_completion(&frame), Ok(message));
}

fn slot() -> SlotRef {
    SlotRef {
        region_sequence: 4,
        region_size: 1024,
        slot_index: 2,
        offset: 128,
        capacity: 64,
        generation: 9,
    }
}

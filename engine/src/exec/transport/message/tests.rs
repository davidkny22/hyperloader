use super::{decode_completion, decode_dispatch, encode_completion, encode_dispatch};
use crate::arena::SlotRef;
use crate::exec::{CompletionMessage, CompletionStatus, DispatchMessage};

#[test]
fn dispatch_decoder_rejects_corrupted_header_and_reserved_bytes() {
    let message = DispatchMessage {
        position: 8,
        stage_plan: 3,
        index: 19,
        worker: 2,
        slot: slot(),
        exception_slot: SlotRef {
            slot_index: 3,
            ..slot()
        },
    };
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

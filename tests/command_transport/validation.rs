use super::common::{registry, slot};
use _hyperloader::arena::{RegionToken, SlotRef};
use _hyperloader::exec::{
    CommandTransport, CompletionMessage, CompletionStatus, DispatchMessage, ExceptionRef,
    TransportError,
};

#[test]
fn message_validation_rejects_inconsistent_fields() {
    let token = RegionToken::random().expect("transport token");
    let transport =
        CommandTransport::create(registry("validation"), token, 21, 4, 4).expect("transport");
    let invalid_slot = SlotRef {
        capacity: 0,
        ..slot(0)
    };
    assert!(matches!(
        transport.try_send_dispatch(DispatchMessage {
            position: 0,
            stage_plan: 0,
            index: 0,
            worker: 0,
            slot: invalid_slot,
            exception_slot: slot(1),
        }),
        Err(TransportError::InvalidMessage("slot reference"))
    ));
    assert!(matches!(
        transport.try_send_completion(CompletionMessage {
            position: 0,
            worker: 0,
            status: CompletionStatus::Ready,
            slot: slot(0),
            produced_length: 1,
            exception: Some(ExceptionRef {
                slot: slot(1),
                length: 1,
            }),
        }),
        Err(TransportError::InvalidMessage("completion side slab"))
    ));
    assert!(CommandTransport::attach(token, 21, 8, 4).is_err());
}

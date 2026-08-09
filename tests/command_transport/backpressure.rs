use super::common::{dispatch, ready_completion, transport};
use _hyperloader::exec::TransportError;

#[test]
fn dispatch_preserves_fifo_order_and_reports_backpressure() {
    let transport = transport("fifo", 20, 2, 2);
    let first = dispatch(1, 3);
    let second = dispatch(2, 3);
    transport
        .try_send_dispatch(first)
        .expect("enqueue first dispatch");
    transport
        .try_send_dispatch(second)
        .expect("enqueue second dispatch");
    assert!(matches!(
        transport.try_send_dispatch(dispatch(3, 3)),
        Err(TransportError::DispatchFull)
    ));
    assert_eq!(transport.try_recv_dispatch().expect("receive first"), first);
    assert_eq!(
        transport.try_recv_dispatch().expect("receive second"),
        second
    );
    assert!(matches!(
        transport.try_recv_dispatch(),
        Err(TransportError::DispatchEmpty)
    ));

    let first_completion = ready_completion(10, 3);
    let second_completion = ready_completion(11, 3);
    transport
        .try_send_completion(first_completion)
        .expect("enqueue first completion");
    transport
        .try_send_completion(second_completion)
        .expect("enqueue second completion");
    assert!(matches!(
        transport.try_send_completion(ready_completion(12, 3)),
        Err(TransportError::CompletionFull)
    ));
    assert_eq!(
        transport
            .try_recv_completion()
            .expect("receive first completion"),
        first_completion
    );
    assert_eq!(
        transport
            .try_recv_completion()
            .expect("receive second completion"),
        second_completion
    );
    assert!(matches!(
        transport.try_recv_completion(),
        Err(TransportError::CompletionEmpty)
    ));

    for position in 0..128 {
        let message = dispatch(position, 9);
        transport
            .try_send_dispatch(message)
            .expect("enqueue wraparound dispatch");
        assert_eq!(
            transport
                .try_recv_dispatch()
                .expect("receive wraparound dispatch"),
            message
        );
    }
}

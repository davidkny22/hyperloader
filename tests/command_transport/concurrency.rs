use super::common::{slot, transport};
use _hyperloader::exec::{CompletionMessage, CompletionStatus, TransportError};
use std::collections::BTreeSet;
use std::sync::Arc;

#[test]
fn concurrent_producers_deliver_each_completion_once() {
    let transport = Arc::new(transport("concurrent", 22, 8, 64));
    let producers: Vec<_> = (0_u64..4)
        .map(|worker| {
            let transport = Arc::clone(&transport);
            std::thread::spawn(move || {
                for ordinal in 0_u64..64 {
                    let position = (worker << 32) | ordinal;
                    let message = CompletionMessage {
                        position,
                        worker: worker as u32,
                        status: CompletionStatus::Ready,
                        slot: slot(position),
                        produced_length: 8,
                        exception: None,
                    };
                    loop {
                        match transport.try_send_completion(message) {
                            Ok(()) => break,
                            Err(TransportError::CompletionFull) => std::thread::yield_now(),
                            Err(error) => panic!("unexpected completion failure: {error}"),
                        }
                    }
                }
            })
        })
        .collect();

    let mut received = BTreeSet::new();
    while received.len() < 256 {
        match transport.try_recv_completion() {
            Ok(message) => assert!(received.insert(message.position)),
            Err(TransportError::CompletionEmpty) => std::thread::yield_now(),
            Err(error) => panic!("unexpected receive failure: {error}"),
        }
    }
    for producer in producers {
        producer.join().expect("completion producer");
    }
    assert_eq!(received.len(), 256);
}

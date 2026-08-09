use super::common::{dispatch, registry};
use _hyperloader::arena::RegionToken;
use _hyperloader::exec::{CommandTransport, CompletionMessage, CompletionStatus};
use std::process::Command;

#[test]
fn dispatch_and_completion_cross_an_independent_process() {
    let registry = registry("process");
    let token = RegionToken::random().expect("transport token");
    let transport = CommandTransport::create(registry, token, 23, 4, 4).expect("transport");
    let message = dispatch(91, 12);
    transport
        .try_send_dispatch(message)
        .expect("send child dispatch");
    let encoded: String = token
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let status = Command::new(std::env::current_exe().expect("current test executable"))
        .args(["--exact", "process::command_transport_child", "--ignored"])
        .env("HYPERLOADER_COMMAND_TOKEN", encoded)
        .status()
        .expect("launch command child");
    assert!(status.success());
    assert_eq!(
        transport
            .try_recv_completion()
            .expect("receive child completion"),
        CompletionMessage {
            position: message.position,
            worker: message.worker,
            status: CompletionStatus::Ready,
            slot: message.slot,
            produced_length: 17,
            exception: None,
        }
    );
}

#[test]
#[ignore = "launched by dispatch_and_completion_cross_an_independent_process"]
fn command_transport_child() {
    let encoded = std::env::var("HYPERLOADER_COMMAND_TOKEN").expect("command token");
    assert_eq!(encoded.len(), 32);
    let mut bytes = [0_u8; 16];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&encoded[index * 2..index * 2 + 2], 16).expect("token byte");
    }
    let transport = CommandTransport::attach(RegionToken::from_bytes(bytes), 23, 4, 4)
        .expect("attach command child");
    let received = transport
        .try_recv_dispatch()
        .expect("receive parent dispatch");
    assert_eq!(received, dispatch(91, 12));
    transport
        .try_send_completion(CompletionMessage {
            position: received.position,
            worker: received.worker,
            status: CompletionStatus::Ready,
            slot: received.slot,
            produced_length: 17,
            exception: None,
        })
        .expect("send parent completion");
}

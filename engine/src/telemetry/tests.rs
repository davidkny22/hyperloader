use super::{ControllerRecord, INSTRUMENTS, Telemetry};

#[test]
fn registry_ids_are_unique_and_versioned() {
    let mut ids = INSTRUMENTS.iter().map(|item| item.id).collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), INSTRUMENTS.len());
    assert!(INSTRUMENTS.iter().all(|item| item.version == 1));
}

#[test]
fn snapshot_carries_tails_decisions_and_epoch_reset() {
    let telemetry = Telemetry::new();
    telemetry.record_startup(90);
    for latency in [10, 20, 30, 40] {
        telemetry.record_delivery(2, 16, latency, 100);
    }
    telemetry.record_stall();
    telemetry.record_gil_restore();
    telemetry.record_controller(ControllerRecord {
        previous_width: 2,
        width: 1,
        reason: "bandwidth-ceiling".to_owned(),
        starvation: true,
        binding: Some("bandwidth".to_owned()),
        resource_loss: 0.02,
    });

    let current = telemetry.snapshot();
    assert_eq!(current.startup_ns, 90);
    assert_eq!(current.current.delivered_samples, 8);
    assert_eq!(current.current.delivery_latency_ns, [31, 63, 63]);
    assert_eq!(current.current.ceiling_binds, 1);
    assert_eq!(current.current.gil_restore_events, 1);

    telemetry.finish_epoch(7);
    let finished = telemetry.snapshot();
    assert_eq!(finished.current.delivered_samples, 0);
    assert_eq!(finished.current.gil_restore_events, 0);
    assert_eq!(finished.last_epoch.expect("completed epoch").epoch, 7);
}

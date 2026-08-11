use super::NativeSlot;

#[test]
fn native_slot_can_cross_the_batch_future_boundary() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<NativeSlot>();
}

//! Refcounted consumer views over immutable delivered slots.

use super::slab::{ArenaInner, locate_slot, recycle};
use super::{ArenaError, SlotRef, SlotState};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

/// One consumer-held reference whose final drop recycles the slot.
pub struct DeliveryView {
    pub(super) inner: Arc<Mutex<ArenaInner>>,
    pub(super) slot: SlotRef,
    pub(super) released: bool,
}

impl DeliveryView {
    /// Return the stable slot coordinates backing this view.
    pub const fn slot(&self) -> SlotRef {
        self.slot
    }

    /// Copy the immutable delivered bytes for validation or metadata paths.
    pub fn to_vec(&self) -> Result<Vec<u8>, ArenaError> {
        let inner = self.inner.lock().map_err(|_| ArenaError::Synchronization)?;
        let (slab_index, slot_index) = locate_slot(&inner.slabs, self.slot)?;
        let length = match inner.slabs[slab_index].slots[slot_index].state {
            SlotState::Delivered { length, .. } => length,
            actual => {
                return Err(ArenaError::InvalidTransition {
                    expected: "delivered",
                    actual,
                });
            }
        };
        let start = self.slot.offset as usize;
        let end = start + length;
        // SAFETY: delivered slots are immutable until every view drops, and this view holds one
        // reference while the copy is made under the metadata lock.
        Ok(unsafe { inner.slabs[slab_index].region.payload()[start..end].to_vec() })
    }

    /// Return a stable zero-copy pointer and length for a foreign storage wrapper.
    ///
    /// # Safety
    ///
    /// The caller must not outlive this view and must not mutate the returned bytes. The foreign
    /// storage deleter must own and drop this `DeliveryView` exactly once.
    pub unsafe fn as_ptr_len(&self) -> Result<(NonNull<u8>, usize), ArenaError> {
        let inner = self.inner.lock().map_err(|_| ArenaError::Synchronization)?;
        let (slab_index, slot_index) = locate_slot(&inner.slabs, self.slot)?;
        let length = match inner.slabs[slab_index].slots[slot_index].state {
            SlotState::Delivered { length, .. } => length,
            actual => {
                return Err(ArenaError::InvalidTransition {
                    expected: "delivered",
                    actual,
                });
            }
        };
        // SAFETY: the mapping remains alive through `inner`, the slot is delivered and immutable,
        // and the checked slot offset lies within its region payload.
        let pointer = unsafe {
            inner.slabs[slab_index]
                .region
                .payload()
                .as_ptr()
                .add(self.slot.offset as usize)
                .cast_mut()
        };
        Ok((
            NonNull::new(pointer).expect("mapped payload pointer is non-null"),
            length,
        ))
    }

    fn release(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        let Ok((slab_index, slot_index)) = locate_slot(&inner.slabs, self.slot) else {
            return;
        };
        let meta = &mut inner.slabs[slab_index].slots[slot_index];
        let SlotState::Delivered { remaining, length } = meta.state else {
            return;
        };
        if remaining == 1 {
            recycle(meta);
        } else {
            meta.state = SlotState::Delivered {
                remaining: remaining - 1,
                length,
            };
        }
    }
}

impl Drop for DeliveryView {
    fn drop(&mut self) {
        self.release();
    }
}

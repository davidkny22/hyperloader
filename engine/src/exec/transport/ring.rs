//! Bounded MPMC queue ownership over cache-line-aligned shared slots.

#[cfg(not(target_has_atomic = "64"))]
compile_error!("the command transport requires native 64-bit atomics");

use super::message::FRAME_SIZE;
use std::cell::UnsafeCell;
use std::hint::spin_loop;
use std::sync::atomic::{AtomicU64, Ordering, fence};

pub(super) const CACHE_LINE: usize = 64;

#[repr(C, align(64))]
pub(super) struct RingHeader {
    enqueue: AtomicU64,
    enqueue_padding: [u8; 56],
    dequeue: AtomicU64,
    dequeue_padding: [u8; 56],
}

impl RingHeader {
    pub(super) const fn new() -> Self {
        Self {
            enqueue: AtomicU64::new(0),
            enqueue_padding: [0; 56],
            dequeue: AtomicU64::new(0),
            dequeue_padding: [0; 56],
        }
    }
}

#[repr(C, align(64))]
pub(super) struct RingSlot {
    sequence: AtomicU64,
    frame: UnsafeCell<[u8; FRAME_SIZE]>,
}

impl RingSlot {
    pub(super) const fn new(sequence: u64) -> Self {
        Self {
            sequence: AtomicU64::new(sequence),
            frame: UnsafeCell::new([0; FRAME_SIZE]),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct QueueView {
    header: *const RingHeader,
    slots: *const RingSlot,
    capacity: u64,
    mask: u64,
}

impl QueueView {
    /// Construct a view after layout validation.
    ///
    /// # Safety
    ///
    /// Both pointers must be aligned, initialized, live for the view's use, and address one
    /// header followed by at least `capacity` slots in a shared mapping.
    pub(super) unsafe fn new(header: *mut u8, capacity: usize) -> Self {
        let slots = header
            .wrapping_add(size_of::<RingHeader>())
            .cast::<RingSlot>();
        Self {
            header: header.cast::<RingHeader>(),
            slots,
            capacity: capacity as u64,
            mask: capacity as u64 - 1,
        }
    }

    pub(super) fn try_push(&self, frame: [u8; FRAME_SIZE]) -> Result<(), ()> {
        // SAFETY: construction validated the mapped header and its lifetime is held by the
        // transport. All header fields are atomic.
        let header = unsafe { &*self.header };
        let mut position = header.enqueue.load(Ordering::Relaxed);
        let mut spins = 0_u8;
        loop {
            // SAFETY: masking by capacity selects one initialized slot.
            let slot = unsafe { &*self.slots.add((position & self.mask) as usize) };
            let sequence = slot.sequence.load(Ordering::Acquire);
            let difference = sequence.wrapping_sub(position) as i64;
            if difference == 0 {
                match header.enqueue.compare_exchange_weak(
                    position,
                    position.wrapping_add(1),
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // SAFETY: this producer exclusively owns the claimed generation until
                        // the release store publishes it to a consumer.
                        unsafe { *slot.frame.get() = frame };
                        slot.sequence
                            .store(position.wrapping_add(1), Ordering::Release);
                        return Ok(());
                    }
                    Err(observed) => position = observed,
                }
            } else if difference < 0 {
                fence(Ordering::SeqCst);
                if header
                    .dequeue
                    .load(Ordering::Relaxed)
                    .wrapping_add(self.capacity)
                    == position
                {
                    return Err(());
                }
                position = header.enqueue.load(Ordering::Relaxed);
            } else {
                position = header.enqueue.load(Ordering::Relaxed);
            }
            backoff(&mut spins);
        }
    }

    pub(super) fn try_pop(&self) -> Option<[u8; FRAME_SIZE]> {
        // SAFETY: construction validated the mapped header and its lifetime is held by the
        // transport. All header fields are atomic.
        let header = unsafe { &*self.header };
        let mut position = header.dequeue.load(Ordering::Relaxed);
        let mut spins = 0_u8;
        loop {
            // SAFETY: masking by capacity selects one initialized slot.
            let slot = unsafe { &*self.slots.add((position & self.mask) as usize) };
            let sequence = slot.sequence.load(Ordering::Acquire);
            let expected = position.wrapping_add(1);
            let difference = sequence.wrapping_sub(expected) as i64;
            if difference == 0 {
                match header.dequeue.compare_exchange_weak(
                    position,
                    position.wrapping_add(1),
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // SAFETY: this consumer exclusively owns the claimed generation after
                        // observing the producer's release store.
                        let frame = unsafe { *slot.frame.get() };
                        slot.sequence
                            .store(position.wrapping_add(self.capacity), Ordering::Release);
                        return Some(frame);
                    }
                    Err(observed) => position = observed,
                }
            } else if difference < 0 {
                fence(Ordering::SeqCst);
                if header.enqueue.load(Ordering::Relaxed) == position {
                    return None;
                }
                position = header.dequeue.load(Ordering::Relaxed);
            } else {
                position = header.dequeue.load(Ordering::Relaxed);
            }
            backoff(&mut spins);
        }
    }
}

fn backoff(spins: &mut u8) {
    if *spins < 8 {
        spin_loop();
        *spins += 1;
    } else {
        std::thread::yield_now();
    }
}

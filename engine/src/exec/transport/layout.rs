//! Checked shared-memory layout, initialization, and attachment validation.

use super::TransportError;
use super::ring::{CACHE_LINE, QueueView, RingHeader, RingSlot};
use crate::arena::NamedRegion;
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};

const MAGIC: &[u8; 8] = b"HLCMDRNG";
const VERSION: u32 = 1;
const READY: u32 = 1;

#[repr(C, align(64))]
struct TransportHeader {
    magic: [u8; 8],
    version: u32,
    ready: AtomicU32,
    dispatch_capacity: u32,
    completion_capacity: u32,
    payload_size: u64,
    dispatch_offset: u64,
    completion_offset: u64,
    reserved: [u8; 80],
}

impl TransportHeader {
    fn new(layout: SharedLayout) -> Self {
        Self {
            magic: *MAGIC,
            version: VERSION,
            ready: AtomicU32::new(0),
            dispatch_capacity: layout.dispatch_capacity as u32,
            completion_capacity: layout.completion_capacity as u32,
            payload_size: layout.payload_size as u64,
            dispatch_offset: layout.dispatch_offset as u64,
            completion_offset: layout.completion_offset as u64,
            reserved: [0; 80],
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct SharedLayout {
    dispatch_capacity: usize,
    completion_capacity: usize,
    dispatch_offset: usize,
    completion_offset: usize,
    layout_size: usize,
    payload_size: usize,
}

impl SharedLayout {
    pub(super) fn new(
        dispatch_capacity: usize,
        completion_capacity: usize,
    ) -> Result<Self, TransportError> {
        validate_capacity(dispatch_capacity)?;
        validate_capacity(completion_capacity)?;
        let dispatch_offset = size_of::<TransportHeader>();
        let dispatch_size = queue_size(dispatch_capacity)?;
        let completion_offset = dispatch_offset
            .checked_add(dispatch_size)
            .ok_or(TransportError::LayoutOverflow)?;
        let layout_size = completion_offset
            .checked_add(queue_size(completion_capacity)?)
            .ok_or(TransportError::LayoutOverflow)?;
        let payload_size = layout_size
            .checked_add(CACHE_LINE - 1)
            .ok_or(TransportError::LayoutOverflow)?;
        Ok(Self {
            dispatch_capacity,
            completion_capacity,
            dispatch_offset,
            completion_offset,
            layout_size,
            payload_size,
        })
    }

    pub(super) const fn payload_size(self) -> usize {
        self.payload_size
    }
}

pub(super) unsafe fn initialize(
    region: &NamedRegion,
    layout: SharedLayout,
) -> Result<(QueueView, QueueView), TransportError> {
    let base = aligned_base(region, layout)?;
    // SAFETY: creation owns this mapping exclusively and every address was checked and aligned.
    unsafe {
        ptr::write(base.cast::<TransportHeader>(), TransportHeader::new(layout));
        initialize_queue(base.add(layout.dispatch_offset), layout.dispatch_capacity);
        initialize_queue(
            base.add(layout.completion_offset),
            layout.completion_capacity,
        );
        (*base.cast::<TransportHeader>())
            .ready
            .store(READY, Ordering::Release);
        Ok((
            QueueView::new(base.add(layout.dispatch_offset), layout.dispatch_capacity),
            QueueView::new(
                base.add(layout.completion_offset),
                layout.completion_capacity,
            ),
        ))
    }
}

pub(super) unsafe fn validate(
    region: &NamedRegion,
    layout: SharedLayout,
) -> Result<(QueueView, QueueView), TransportError> {
    let base = aligned_base(region, layout)?;
    // SAFETY: the aligned header lies within a mapping validated to the computed payload size.
    let header = unsafe { &*base.cast::<TransportHeader>() };
    if header.ready.load(Ordering::Acquire) != READY {
        return Err(TransportError::HeaderMismatch("publication state"));
    }
    if &header.magic != MAGIC {
        return Err(TransportError::HeaderMismatch("magic"));
    }
    if header.version != VERSION {
        return Err(TransportError::HeaderMismatch("version"));
    }
    if header.dispatch_capacity != layout.dispatch_capacity as u32
        || header.completion_capacity != layout.completion_capacity as u32
    {
        return Err(TransportError::HeaderMismatch("capacities"));
    }
    if header.payload_size != layout.payload_size as u64
        || header.dispatch_offset != layout.dispatch_offset as u64
        || header.completion_offset != layout.completion_offset as u64
        || header.reserved.iter().any(|byte| *byte != 0)
    {
        return Err(TransportError::HeaderMismatch("layout"));
    }
    // SAFETY: every persisted field matches the checked local layout.
    unsafe {
        Ok((
            QueueView::new(base.add(layout.dispatch_offset), layout.dispatch_capacity),
            QueueView::new(
                base.add(layout.completion_offset),
                layout.completion_capacity,
            ),
        ))
    }
}

fn aligned_base(region: &NamedRegion, layout: SharedLayout) -> Result<*mut u8, TransportError> {
    let payload = region.payload_ptr();
    let padding = payload.align_offset(CACHE_LINE);
    if padding == usize::MAX
        || padding
            .checked_add(layout.layout_size)
            .is_none_or(|used| used > region.payload_size())
    {
        return Err(TransportError::HeaderMismatch("alignment"));
    }
    Ok(payload.wrapping_add(padding))
}

unsafe fn initialize_queue(base: *mut u8, capacity: usize) {
    // SAFETY: the caller provides an aligned, exclusive queue range of the checked size.
    unsafe {
        ptr::write(base.cast::<RingHeader>(), RingHeader::new());
        let slots = base.add(size_of::<RingHeader>()).cast::<RingSlot>();
        for index in 0..capacity {
            ptr::write(slots.add(index), RingSlot::new(index as u64));
        }
    }
}

fn validate_capacity(capacity: usize) -> Result<(), TransportError> {
    if capacity < 2 || !capacity.is_power_of_two() || u32::try_from(capacity).is_err() {
        return Err(TransportError::InvalidCapacity(capacity));
    }
    Ok(())
}

fn queue_size(capacity: usize) -> Result<usize, TransportError> {
    size_of::<RingSlot>()
        .checked_mul(capacity)
        .and_then(|slots| size_of::<RingHeader>().checked_add(slots))
        .ok_or(TransportError::LayoutOverflow)
}

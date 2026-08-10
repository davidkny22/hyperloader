//! Worker-local attachment cache and exclusive slot writes.

use crate::arena::{NamedRegion, RegionError, RegionToken, SlotRef};
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// A worker-local cache of validated arena-region attachments.
pub struct ArenaWriter {
    token: RegionToken,
    regions: HashMap<u16, NamedRegion>,
}

impl ArenaWriter {
    /// Create an empty attachment cache for one loader identity.
    pub fn new(token: RegionToken) -> Self {
        Self {
            token,
            regions: HashMap::new(),
        }
    }

    /// Write one payload into a slot exclusively assigned to this worker.
    pub fn write(&mut self, slot: SlotRef, payload: &[u8]) -> Result<(), ArenaWriterError> {
        self.write_at(slot, 0, payload)
    }

    /// Write one row at a byte offset inside an exclusively assigned batch slot.
    pub fn write_at(
        &mut self,
        slot: SlotRef,
        relative_offset: usize,
        payload: &[u8],
    ) -> Result<(), ArenaWriterError> {
        validate_slot(slot, relative_offset, payload.len())?;
        let payload_size = usize::try_from(slot.region_size)
            .map_err(|_| ArenaWriterError::InvalidSlot("region size"))?;
        if !self.regions.contains_key(&slot.region_sequence) {
            let region = NamedRegion::attach(self.token, slot.region_sequence, payload_size)?;
            self.regions.insert(slot.region_sequence, region);
        }
        let region = self
            .regions
            .get_mut(&slot.region_sequence)
            .expect("region was attached above");
        if region.payload_size() != payload_size {
            return Err(ArenaWriterError::InvalidSlot("cached region size"));
        }
        let start = usize::try_from(slot.offset)
            .map_err(|_| ArenaWriterError::InvalidSlot("slot offset"))?
            .checked_add(relative_offset)
            .ok_or(ArenaWriterError::InvalidSlot("slot range"))?;
        let end = start
            .checked_add(payload.len())
            .ok_or(ArenaWriterError::InvalidSlot("slot range"))?;
        // SAFETY: the dispatch protocol assigns this slot exclusively to the worker until its
        // completion is published. Validation bounds the write within the attached region.
        unsafe { region.payload_mut()[start..end].copy_from_slice(payload) };
        Ok(())
    }
}

/// A validated attachment or worker-write failure.
#[derive(Debug)]
pub enum ArenaWriterError {
    /// The slot coordinates cannot describe a bounded region write.
    InvalidSlot(&'static str),
    /// The encoded payload exceeds the reserved capacity.
    SlotOverflow { capacity: u64, actual: usize },
    /// Named-region attachment failed.
    Region(RegionError),
}

impl Display for ArenaWriterError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSlot(field) => write!(formatter, "arena slot has an invalid {field}"),
            Self::SlotOverflow { capacity, actual } => write!(
                formatter,
                "arena slot capacity is {capacity} bytes but payload is {actual} bytes"
            ),
            Self::Region(source) => write!(formatter, "arena worker attachment failed: {source}"),
        }
    }
}

impl Error for ArenaWriterError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Region(source) => Some(source),
            _ => None,
        }
    }
}

impl From<RegionError> for ArenaWriterError {
    fn from(error: RegionError) -> Self {
        Self::Region(error)
    }
}

fn validate_slot(
    slot: SlotRef,
    relative_offset: usize,
    payload_length: usize,
) -> Result<(), ArenaWriterError> {
    let end = slot
        .offset
        .checked_add(slot.capacity)
        .ok_or(ArenaWriterError::InvalidSlot("slot range"))?;
    if slot.region_size == 0 || slot.capacity == 0 || end > slot.region_size {
        return Err(ArenaWriterError::InvalidSlot("slot range"));
    }
    let produced = relative_offset
        .checked_add(payload_length)
        .ok_or(ArenaWriterError::InvalidSlot("slot range"))?;
    if produced as u64 > slot.capacity {
        return Err(ArenaWriterError::SlotOverflow {
            capacity: slot.capacity,
            actual: produced,
        });
    }
    Ok(())
}

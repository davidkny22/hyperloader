//! Explicit fixed-width command encoding independent of Rust ABI layout.

use super::TransportError;
use crate::arena::SlotRef;

pub(super) const FRAME_SIZE: usize = 128;
const MAGIC: &[u8; 4] = b"HLCM";
const VERSION: u8 = 1;
const DISPATCH_KIND: u8 = 1;
const COMPLETION_KIND: u8 = 2;
const READY_STATUS: u8 = 1;
const EXCEPTION_STATUS: u8 = 2;
const SIDE_PRESENT: u8 = 1;

/// One scheduler-to-worker command.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DispatchMessage {
    /// Sampler-stream position owned by this command.
    pub position: u64,
    /// Loader epoch used by per-sample RNG derivation.
    pub epoch: u64,
    /// Stable identifier for the stage plan to execute.
    pub stage_plan: u32,
    /// Dataset or sampler index consumed by the black-box stage.
    pub index: u64,
    /// Worker route selected by the executor.
    pub worker: u32,
    /// This position closes its default-collation batch.
    pub batch_end: bool,
    /// Arena slot receiving the produced payload.
    pub slot: SlotRef,
    /// Arena side slot reserved for an exception payload.
    pub exception_slot: SlotRef,
}

/// Completion outcome encoded in the fixed command frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionStatus {
    /// The primary slot contains a produced payload.
    Ready,
    /// User code raised and the exception payload occupies the side slab.
    Exception,
}

/// One exception payload stored outside the fixed command ring.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExceptionRef {
    /// Arena slot containing the encoded exception type and traceback.
    pub slot: SlotRef,
    /// Exact encoded byte length within the slot.
    pub length: u64,
}

/// One worker-to-delivery completion command.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompletionMessage {
    /// Sampler-stream position whose work terminated.
    pub position: u64,
    /// Worker that executed or failed the command.
    pub worker: u32,
    /// Ready or exception outcome.
    pub status: CompletionStatus,
    /// Primary sample or batch slot from the dispatch command.
    pub slot: SlotRef,
    /// Exact produced byte length, zero on exception.
    pub produced_length: u64,
    /// Side-slab reference required for an exception and absent on success.
    pub exception: Option<ExceptionRef>,
}

pub(super) fn encode_dispatch(
    message: DispatchMessage,
) -> Result<[u8; FRAME_SIZE], TransportError> {
    validate_slot(message.slot).map_err(TransportError::InvalidMessage)?;
    let mut frame = frame_header(DISPATCH_KIND, 0, 0, message.position, message.worker);
    put_u32(&mut frame, 20, message.stage_plan);
    put_u64(&mut frame, 24, message.index);
    put_slot(&mut frame, 32, message.slot);
    validate_slot(message.exception_slot).map_err(TransportError::InvalidMessage)?;
    put_slot(&mut frame, 72, message.exception_slot);
    put_u64(&mut frame, 112, message.epoch);
    frame[120] = u8::from(message.batch_end);
    Ok(frame)
}

pub(super) fn decode_dispatch(frame: &[u8; FRAME_SIZE]) -> Result<DispatchMessage, &'static str> {
    validate_header(frame, DISPATCH_KIND)?;
    if frame[6] != 0 || frame[7] != 0 || frame[121..].iter().any(|byte| *byte != 0) {
        return Err("dispatch reserved fields");
    }
    let batch_end = match frame[120] {
        0 => false,
        1 => true,
        _ => return Err("dispatch batch end"),
    };
    let slot = get_slot(frame, 32)?;
    Ok(DispatchMessage {
        position: get_u64(frame, 8),
        epoch: get_u64(frame, 112),
        worker: get_u32(frame, 16),
        stage_plan: get_u32(frame, 20),
        index: get_u64(frame, 24),
        batch_end,
        slot,
        exception_slot: get_slot(frame, 72)?,
    })
}

pub(super) fn encode_completion(
    message: CompletionMessage,
) -> Result<[u8; FRAME_SIZE], TransportError> {
    validate_slot(message.slot).map_err(TransportError::InvalidMessage)?;
    if message.produced_length > message.slot.capacity {
        return Err(TransportError::InvalidMessage("produced length"));
    }
    let (status, flags) = match (message.status, message.exception) {
        (CompletionStatus::Ready, None) => (READY_STATUS, 0),
        (CompletionStatus::Exception, Some(exception)) => {
            validate_exception(exception).map_err(TransportError::InvalidMessage)?;
            if message.produced_length != 0 {
                return Err(TransportError::InvalidMessage("exception produced length"));
            }
            (EXCEPTION_STATUS, SIDE_PRESENT)
        }
        _ => return Err(TransportError::InvalidMessage("completion side slab")),
    };
    let mut frame = frame_header(
        COMPLETION_KIND,
        status,
        flags,
        message.position,
        message.worker,
    );
    put_slot(&mut frame, 32, message.slot);
    put_u64(&mut frame, 72, message.produced_length);
    if let Some(exception) = message.exception {
        put_slot(&mut frame, 80, exception.slot);
        put_u64(&mut frame, 120, exception.length);
    }
    Ok(frame)
}

pub(super) fn decode_completion(
    frame: &[u8; FRAME_SIZE],
) -> Result<CompletionMessage, &'static str> {
    validate_header(frame, COMPLETION_KIND)?;
    if frame[20..32].iter().any(|byte| *byte != 0) {
        return Err("completion reserved fields");
    }
    let slot = get_slot(frame, 32)?;
    let produced_length = get_u64(frame, 72);
    if produced_length > slot.capacity {
        return Err("produced length");
    }
    let (status, exception) = match (frame[6], frame[7]) {
        (READY_STATUS, 0) => {
            if frame[80..].iter().any(|byte| *byte != 0) {
                return Err("ready side slab");
            }
            (CompletionStatus::Ready, None)
        }
        (EXCEPTION_STATUS, SIDE_PRESENT) => {
            if produced_length != 0 {
                return Err("exception produced length");
            }
            let exception = ExceptionRef {
                slot: get_slot(frame, 80)?,
                length: get_u64(frame, 120),
            };
            validate_exception(exception)?;
            (CompletionStatus::Exception, Some(exception))
        }
        _ => return Err("completion status"),
    };
    Ok(CompletionMessage {
        position: get_u64(frame, 8),
        worker: get_u32(frame, 16),
        status,
        slot,
        produced_length,
        exception,
    })
}

fn frame_header(kind: u8, status: u8, flags: u8, position: u64, worker: u32) -> [u8; FRAME_SIZE] {
    let mut frame = [0_u8; FRAME_SIZE];
    frame[..4].copy_from_slice(MAGIC);
    frame[4] = VERSION;
    frame[5] = kind;
    frame[6] = status;
    frame[7] = flags;
    put_u64(&mut frame, 8, position);
    put_u32(&mut frame, 16, worker);
    frame
}

fn validate_header(frame: &[u8; FRAME_SIZE], kind: u8) -> Result<(), &'static str> {
    if &frame[..4] != MAGIC {
        return Err("magic");
    }
    if frame[4] != VERSION {
        return Err("version");
    }
    if frame[5] != kind {
        return Err("kind");
    }
    Ok(())
}

fn validate_exception(exception: ExceptionRef) -> Result<(), &'static str> {
    validate_slot(exception.slot)?;
    if exception.length == 0 || exception.length > exception.slot.capacity {
        return Err("exception length");
    }
    Ok(())
}

fn validate_slot(slot: SlotRef) -> Result<(), &'static str> {
    if slot.region_size == 0
        || slot.capacity == 0
        || slot
            .offset
            .checked_add(slot.capacity)
            .is_none_or(|end| end > slot.region_size)
    {
        return Err("slot reference");
    }
    Ok(())
}

fn put_slot(frame: &mut [u8; FRAME_SIZE], offset: usize, slot: SlotRef) {
    put_u16(frame, offset, slot.region_sequence);
    put_u32(frame, offset + 4, slot.slot_index);
    put_u64(frame, offset + 8, slot.offset);
    put_u64(frame, offset + 16, slot.capacity);
    put_u64(frame, offset + 24, slot.generation);
    put_u64(frame, offset + 32, slot.region_size);
}

fn get_slot(frame: &[u8; FRAME_SIZE], offset: usize) -> Result<SlotRef, &'static str> {
    if get_u16(frame, offset + 2) != 0 {
        return Err("slot reserved field");
    }
    let slot = SlotRef {
        region_sequence: get_u16(frame, offset),
        region_size: get_u64(frame, offset + 32),
        slot_index: get_u32(frame, offset + 4),
        offset: get_u64(frame, offset + 8),
        capacity: get_u64(frame, offset + 16),
        generation: get_u64(frame, offset + 24),
    };
    validate_slot(slot)?;
    Ok(slot)
}

fn put_u16(frame: &mut [u8; FRAME_SIZE], offset: usize, value: u16) {
    frame[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(frame: &mut [u8; FRAME_SIZE], offset: usize, value: u32) {
    frame[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(frame: &mut [u8; FRAME_SIZE], offset: usize, value: u64) {
    frame[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u16(frame: &[u8; FRAME_SIZE], offset: usize) -> u16 {
    u16::from_le_bytes(
        frame[offset..offset + 2]
            .try_into()
            .expect("fixed u16 field"),
    )
}

fn get_u32(frame: &[u8; FRAME_SIZE], offset: usize) -> u32 {
    u32::from_le_bytes(
        frame[offset..offset + 4]
            .try_into()
            .expect("fixed u32 field"),
    )
}

fn get_u64(frame: &[u8; FRAME_SIZE], offset: usize) -> u64 {
    u64::from_le_bytes(
        frame[offset..offset + 8]
            .try_into()
            .expect("fixed u64 field"),
    )
}

#[cfg(test)]
mod tests;

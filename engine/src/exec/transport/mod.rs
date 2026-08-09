//! Fixed-message dispatch and completion transport over named shared memory.

mod error;
mod layout;
mod message;
mod ring;

pub use error::TransportError;
pub use message::{CompletionMessage, CompletionStatus, DispatchMessage, ExceptionRef};

use crate::arena::{NamedRegion, RegionName, RegionRegistry, RegionToken};
use layout::{SharedLayout, initialize, validate};
use message::{decode_completion, decode_dispatch, encode_completion, encode_dispatch};
use ring::QueueView;

/// One shared region containing independent dispatch and completion queues.
pub struct CommandTransport {
    region: NamedRegion,
    owner_registry: Option<RegionRegistry>,
    dispatch: QueueView,
    completion: QueueView,
    dispatch_capacity: usize,
    completion_capacity: usize,
}

impl CommandTransport {
    /// Create, register, and initialize a transport region.
    pub fn create(
        registry: RegionRegistry,
        token: RegionToken,
        sequence: u16,
        dispatch_capacity: usize,
        completion_capacity: usize,
    ) -> Result<Self, TransportError> {
        let layout = SharedLayout::new(dispatch_capacity, completion_capacity)?;
        let (region, _) = registry.create_region(token, sequence, layout.payload_size())?;
        // SAFETY: this process exclusively owns a freshly created region, and `layout`
        // bounds and aligns every initialized object within its payload.
        let (dispatch, completion) = match unsafe { initialize(&region, layout) } {
            Ok(queues) => queues,
            Err(error) => {
                let name = region.name().as_str().to_owned();
                let _ = region.unlink();
                let _ = registry.retain(|entry| entry.name != name);
                return Err(error);
            }
        };
        Ok(Self {
            region,
            owner_registry: Some(registry),
            dispatch,
            completion,
            dispatch_capacity,
            completion_capacity,
        })
    }

    /// Attach to an initialized transport after validating its complete layout.
    pub fn attach(
        token: RegionToken,
        sequence: u16,
        dispatch_capacity: usize,
        completion_capacity: usize,
    ) -> Result<Self, TransportError> {
        let layout = SharedLayout::new(dispatch_capacity, completion_capacity)?;
        let region = NamedRegion::attach(token, sequence, layout.payload_size())?;
        // SAFETY: named-region attachment validated identity and size. `validate` checks
        // publication and every layout field before constructing queue views.
        let (dispatch, completion) = unsafe { validate(&region, layout)? };
        Ok(Self {
            region,
            owner_registry: None,
            dispatch,
            completion,
            dispatch_capacity,
            completion_capacity,
        })
    }

    /// Return the portable name used by worker attachment.
    pub fn name(&self) -> &RegionName {
        self.region.name()
    }

    /// Return the fixed capacities of the dispatch and completion queues.
    pub const fn capacities(&self) -> (usize, usize) {
        (self.dispatch_capacity, self.completion_capacity)
    }

    /// Attempt to enqueue one scheduler-to-worker dispatch command.
    pub fn try_send_dispatch(&self, message: DispatchMessage) -> Result<(), TransportError> {
        self.dispatch
            .try_push(encode_dispatch(message)?)
            .map_err(|()| TransportError::DispatchFull)
    }

    /// Attempt to receive one scheduler-to-worker dispatch command.
    pub fn try_recv_dispatch(&self) -> Result<DispatchMessage, TransportError> {
        let frame = self
            .dispatch
            .try_pop()
            .ok_or(TransportError::DispatchEmpty)?;
        decode_dispatch(&frame).map_err(TransportError::InvalidMessage)
    }

    /// Attempt to enqueue one worker-to-delivery completion command.
    pub fn try_send_completion(&self, message: CompletionMessage) -> Result<(), TransportError> {
        self.completion
            .try_push(encode_completion(message)?)
            .map_err(|()| TransportError::CompletionFull)
    }

    /// Attempt to receive one worker-to-delivery completion command.
    pub fn try_recv_completion(&self) -> Result<CompletionMessage, TransportError> {
        let frame = self
            .completion
            .try_pop()
            .ok_or(TransportError::CompletionEmpty)?;
        decode_completion(&frame).map_err(TransportError::InvalidMessage)
    }
}

// SAFETY: the mapping stays alive for every queue view, and queue mutation is restricted to
// atomic position and sequence words plus frames owned through those atomic transitions. Moving
// the owner does not move the operating-system mapping addressed by the queue views.
unsafe impl Send for CommandTransport {}
unsafe impl Sync for CommandTransport {}

impl Drop for CommandTransport {
    fn drop(&mut self) {
        let Some(registry) = &self.owner_registry else {
            return;
        };
        let name = self.region.name().as_str().to_owned();
        let _ = self.region.unlink();
        let _ = registry.retain(|entry| entry.name != name);
    }
}

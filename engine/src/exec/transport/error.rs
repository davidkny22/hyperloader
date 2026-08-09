//! Typed creation, attachment, queue, and wire failures.

use crate::arena::{RegionError, RegistryError};
use std::error::Error;
use std::fmt::{Display, Formatter};

/// A command-transport failure with stable queue and validation categories.
#[derive(Debug)]
pub enum TransportError {
    /// A queue capacity is zero, non-power-of-two, or not representable on the wire.
    InvalidCapacity(usize),
    /// Checked shared-memory layout arithmetic overflowed.
    LayoutOverflow,
    /// Region creation or registry persistence failed.
    Registry(RegistryError),
    /// Named-region attachment failed.
    Region(RegionError),
    /// The mapped transport header is unpublished or inconsistent.
    HeaderMismatch(&'static str),
    /// The bounded dispatch queue has no free slot.
    DispatchFull,
    /// The dispatch queue has no published command.
    DispatchEmpty,
    /// The bounded completion queue has no free slot.
    CompletionFull,
    /// The completion queue has no published command.
    CompletionEmpty,
    /// A command cannot be represented or failed wire validation.
    InvalidMessage(&'static str),
}

impl Display for TransportError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCapacity(capacity) => {
                write!(formatter, "command ring capacity {capacity} is invalid")
            }
            Self::LayoutOverflow => formatter.write_str("command ring layout overflowed"),
            Self::Registry(source) => write!(formatter, "command ring creation failed: {source}"),
            Self::Region(source) => write!(formatter, "command ring attachment failed: {source}"),
            Self::HeaderMismatch(field) => {
                write!(formatter, "command ring header has an invalid {field}")
            }
            Self::DispatchFull => formatter.write_str("dispatch ring is full"),
            Self::DispatchEmpty => formatter.write_str("dispatch ring is empty"),
            Self::CompletionFull => formatter.write_str("completion ring is full"),
            Self::CompletionEmpty => formatter.write_str("completion ring is empty"),
            Self::InvalidMessage(field) => {
                write!(formatter, "command message has an invalid {field}")
            }
        }
    }
}

impl Error for TransportError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Registry(source) => Some(source),
            Self::Region(source) => Some(source),
            _ => None,
        }
    }
}

impl From<RegistryError> for TransportError {
    fn from(error: RegistryError) -> Self {
        Self::Registry(error)
    }
}

impl From<RegionError> for TransportError {
    fn from(error: RegionError) -> Self {
        Self::Region(error)
    }
}

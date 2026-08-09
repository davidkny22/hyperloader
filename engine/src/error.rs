//! Typed failures shared by native engine components.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// Stable categories used when native failures cross component boundaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineErrorKind {
    /// Engine construction or startup failed.
    Initialization,
    /// User configuration violates a declared contract.
    InvalidConfiguration,
    /// A required operating-system resource was unavailable or invalid.
    Resource,
    /// A worker failed while executing assigned work.
    Worker,
    /// An internal invariant was violated.
    Internal,
}

impl Display for EngineErrorKind {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Initialization => "initialization",
            Self::InvalidConfiguration => "invalid configuration",
            Self::Resource => "resource",
            Self::Worker => "worker",
            Self::Internal => "internal",
        };
        formatter.write_str(name)
    }
}

/// A native engine failure with a stable category and an actionable message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineError {
    kind: EngineErrorKind,
    message: String,
}

impl EngineError {
    /// Construct an engine failure without discarding its stable category.
    pub fn new(kind: EngineErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Return the stable category used for boundary-level error mapping.
    pub const fn kind(&self) -> EngineErrorKind {
        self.kind
    }

    /// Return the actionable detail without the category prefix.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for EngineError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.kind, self.message)
    }
}

impl Error for EngineError {}

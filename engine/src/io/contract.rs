//! Stable contracts shared by file-input backends.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io;
use std::str::FromStr;

/// A configured backend preference resolved when a loader plan is built.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendPreference {
    /// Select the native backend for the current platform.
    Auto,
    /// Select Linux io_uring.
    Uring,
    /// Select Windows I/O completion ports.
    Iocp,
    /// Select the positioned-read refuge.
    Pread,
}

impl FromStr for BackendPreference {
    type Err = IoError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "uring" => Ok(Self::Uring),
            "iocp" => Ok(Self::Iocp),
            "pread" => Ok(Self::Pread),
            _ => Err(IoError::UnknownBackend(value.to_owned())),
        }
    }
}

/// The concrete backend selected for one loader plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendKind {
    /// Windows overlapped reads completed through an I/O completion port.
    Iocp,
    /// Portable positioned reads through the standard operating-system file handle.
    Pread,
}

impl BackendKind {
    /// Return the stable configuration spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Iocp => "iocp",
            Self::Pread => "pread",
        }
    }
}

/// One completed range read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReadCompletion {
    bytes_read: usize,
}

impl ReadCompletion {
    pub(crate) const fn new(bytes_read: usize) -> Self {
        Self { bytes_read }
    }

    /// Return the bytes written into the destination before end of file.
    pub const fn bytes_read(self) -> usize {
        self.bytes_read
    }
}

/// A typed file-input failure.
#[derive(Debug)]
pub enum IoError {
    /// The configured backend spelling is not part of the public configuration contract.
    UnknownBackend(String),
    /// The requested backend cannot run on the current target.
    Unavailable {
        /// Requested backend spelling.
        requested: &'static str,
        /// Current Rust target operating system.
        platform: &'static str,
    },
    /// The requested byte count cannot be represented by the platform API.
    InvalidLength(usize),
    /// The byte offset overflowed while completing a short positioned read.
    OffsetOverflow,
    /// An operating-system operation failed.
    Os {
        /// Operation that failed.
        operation: &'static str,
        /// Original operating-system error.
        source: io::Error,
    },
    /// A completion packet did not describe the submitted request.
    CompletionMismatch,
}

impl IoError {
    pub(crate) fn os(operation: &'static str, source: io::Error) -> Self {
        Self::Os { operation, source }
    }
}

impl Display for IoError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownBackend(value) => write!(formatter, "unknown I/O backend {value:?}"),
            Self::Unavailable {
                requested,
                platform,
            } => write!(
                formatter,
                "I/O backend {requested} is unavailable on {platform}"
            ),
            Self::InvalidLength(length) => {
                write!(
                    formatter,
                    "I/O range length {length} exceeds the platform limit"
                )
            }
            Self::OffsetOverflow => formatter.write_str("I/O range offset overflowed"),
            Self::Os { operation, source } => write!(formatter, "{operation} failed: {source}"),
            Self::CompletionMismatch => {
                formatter.write_str("I/O completion did not match the submitted request")
            }
        }
    }
}

impl Error for IoError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Os { source, .. } => Some(source),
            _ => None,
        }
    }
}

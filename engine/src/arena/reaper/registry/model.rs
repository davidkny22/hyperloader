//! Registry records and typed persistence failures.

use crate::arena::{NamedRegion, RegionError, RegionName, RegionToken};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io;

/// One durable ownership record written when a named region is created.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RegistryEntry {
    /// Portable shared-memory name.
    pub name: String,
    /// Complete loader token in lowercase hexadecimal.
    pub token: String,
    /// Creator process identifier.
    pub pid: u32,
    /// Boot identity observed by the creator.
    pub boot_id: String,
    /// Creator process start time in platform-normalized units.
    pub proc_start: u64,
}

impl RegistryEntry {
    /// Build an ownership record from a validated region and process identity.
    pub fn new(
        region: &NamedRegion,
        pid: u32,
        boot_id: impl Into<String>,
        proc_start: u64,
    ) -> Self {
        Self {
            name: region.name().as_str().to_owned(),
            token: region.token().to_hex(),
            pid,
            boot_id: boot_id.into(),
            proc_start,
        }
    }

    pub(crate) fn validated_name(&self) -> Option<RegionName> {
        let token = RegionToken::from_hex(&self.token)?;
        RegionName::from_registry(&self.name, token)
    }
}

/// A recoverable problem found while reading registry evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistryIssue {
    /// One-based line number, or zero for a file-level issue.
    pub line: usize,
    /// Stable issue description for audit logging.
    pub message: String,
}

/// A locked registry read with valid entries and preserved audit issues.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RegistrySnapshot {
    /// Valid, unique ownership records.
    pub entries: Vec<RegistryEntry>,
    /// Corruption or ambiguity that suppresses destructive cleanup.
    pub issues: Vec<RegistryIssue>,
}

/// A typed registry persistence failure.
#[derive(Debug)]
pub enum RegistryError {
    /// A filesystem operation failed.
    Io {
        /// Operation that failed.
        operation: &'static str,
        /// Original filesystem error.
        source: io::Error,
    },
    /// A record could not be serialized.
    Json(serde_json::Error),
    /// A caller attempted to append a malformed or inconsistent record.
    InvalidEntry,
    /// Current process identity could not be established safely.
    Identity(String),
    /// Named-region creation failed before registration.
    Region(RegionError),
}

impl RegistryError {
    pub(super) fn io(operation: &'static str, source: io::Error) -> Self {
        Self::Io { operation, source }
    }
}

impl Display for RegistryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { operation, source } => write!(formatter, "{operation} failed: {source}"),
            Self::Json(source) => write!(formatter, "registry serialization failed: {source}"),
            Self::InvalidEntry => formatter.write_str("registry entry is malformed"),
            Self::Identity(message) => write!(formatter, "process identity failed: {message}"),
            Self::Region(source) => write!(formatter, "named region creation failed: {source}"),
        }
    }
}

impl Error for RegistryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json(source) => Some(source),
            Self::InvalidEntry => None,
            Self::Identity(_) => None,
            Self::Region(source) => Some(source),
        }
    }
}

impl From<serde_json::Error> for RegistryError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

//! Named mapping lifecycle and header validation.

use super::identity::{RegionName, RegionToken, TOKEN_BYTES};
use super::platform;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io;
use std::ops::Range;

const MAGIC: &[u8; 8] = b"HLARENA\0";
const SIZE_BYTES: usize = size_of::<u64>();
const HEADER_LEN: usize = MAGIC.len() + TOKEN_BYTES + SIZE_BYTES;

#[derive(Debug)]
pub enum RegionError {
    /// A region sequence cannot be represented by the fixed name grammar.
    SequenceOutOfRange(u16),
    /// The requested payload cannot be represented by the platform mapping API.
    InvalidSize(usize),
    /// Secure token generation failed.
    Random(String),
    /// Exclusive creation found an existing region with the same name.
    AlreadyExists(RegionName),
    /// No live region has the requested name.
    NotFound(RegionName),
    /// The mapped header does not match the expected identity or size.
    HeaderMismatch(&'static str),
    /// An operating-system operation failed.
    Os {
        /// Operation that failed.
        operation: &'static str,
        /// Original operating-system error.
        source: io::Error,
    },
}

impl RegionError {
    pub(crate) fn os(operation: &'static str, source: io::Error) -> Self {
        Self::Os { operation, source }
    }
}

impl Display for RegionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SequenceOutOfRange(sequence) => {
                write!(
                    formatter,
                    "region sequence {sequence} exceeds the name grammar"
                )
            }
            Self::InvalidSize(size) => write!(formatter, "region payload size {size} is invalid"),
            Self::Random(message) => write!(formatter, "region token generation failed: {message}"),
            Self::AlreadyExists(name) => write!(formatter, "region {name} already exists"),
            Self::NotFound(name) => write!(formatter, "region {name} does not exist"),
            Self::HeaderMismatch(field) => {
                write!(formatter, "region header has an invalid {field}")
            }
            Self::Os { operation, source } => write!(formatter, "{operation} failed: {source}"),
        }
    }
}

impl Error for RegionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Os { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// One full-size shared region whose header was validated before payload access.
pub struct NamedRegion {
    name: RegionName,
    token: RegionToken,
    payload_size: usize,
    pub(super) mapping: platform::Mapping,
}

impl NamedRegion {
    /// Create and initialize a region, failing if its name already exists.
    pub(crate) fn create(
        token: RegionToken,
        sequence: u16,
        payload_size: usize,
    ) -> Result<Self, RegionError> {
        let name = RegionName::new(token, sequence)?;
        let total_size = total_size(payload_size)?;
        let mut mapping = platform::Mapping::create(&name, total_size)?;
        write_header(mapping.as_mut_slice(), token, payload_size);
        Ok(Self {
            name,
            token,
            payload_size,
            mapping,
        })
    }

    /// Attach to a region and reject mismatched or stale mappings before use.
    pub fn attach(
        token: RegionToken,
        sequence: u16,
        payload_size: usize,
    ) -> Result<Self, RegionError> {
        let name = RegionName::new(token, sequence)?;
        let total_size = total_size(payload_size)?;
        let mapping = platform::Mapping::open(&name, total_size)?;
        validate_header(mapping.as_slice(), token, payload_size)?;
        Ok(Self {
            name,
            token,
            payload_size,
            mapping,
        })
    }

    /// Return this region's portable name.
    pub fn name(&self) -> &RegionName {
        &self.name
    }

    /// Return the full identity checked at attachment time.
    pub const fn token(&self) -> RegionToken {
        self.token
    }

    /// Return the payload capacity in bytes.
    pub const fn payload_size(&self) -> usize {
        self.payload_size
    }

    /// Borrow the payload after header validation.
    ///
    /// # Safety
    ///
    /// The caller must hold the arena protocol's read ownership for the complete borrow.
    /// No local or attached process may write the same bytes while the borrow is live.
    pub unsafe fn payload(&self) -> &[u8] {
        &self.mapping.as_slice()[payload_range(self.payload_size)]
    }

    /// Mutably borrow the payload after header validation.
    ///
    /// # Safety
    ///
    /// The caller must hold the arena protocol's exclusive write ownership for the
    /// complete borrow. No local or attached process may access the same bytes.
    pub unsafe fn payload_mut(&mut self) -> &mut [u8] {
        let range = payload_range(self.payload_size);
        &mut self.mapping.as_mut_slice()[range]
    }

    /// Remove the POSIX namespace entry. Windows removes it when all handles close.
    pub fn unlink(&self) -> Result<(), RegionError> {
        platform::unlink(&self.name)
    }
}

pub(crate) fn unlink_registered(name: &RegionName) -> Result<(), RegionError> {
    platform::unlink(name)
}

fn total_size(payload_size: usize) -> Result<usize, RegionError> {
    if payload_size == 0 || u64::try_from(payload_size).is_err() {
        return Err(RegionError::InvalidSize(payload_size));
    }
    HEADER_LEN
        .checked_add(payload_size)
        .ok_or(RegionError::InvalidSize(payload_size))
}

fn payload_range(payload_size: usize) -> Range<usize> {
    HEADER_LEN..HEADER_LEN + payload_size
}

fn write_header(mapping: &mut [u8], token: RegionToken, payload_size: usize) {
    mapping[..MAGIC.len()].copy_from_slice(MAGIC);
    let token_start = MAGIC.len();
    let token_end = token_start + TOKEN_BYTES;
    mapping[token_start..token_end].copy_from_slice(token.as_bytes());
    mapping[token_end..HEADER_LEN].copy_from_slice(&(payload_size as u64).to_le_bytes());
}

fn validate_header(
    mapping: &[u8],
    token: RegionToken,
    payload_size: usize,
) -> Result<(), RegionError> {
    if mapping.get(..MAGIC.len()) != Some(MAGIC) {
        return Err(RegionError::HeaderMismatch("magic"));
    }
    let token_start = MAGIC.len();
    let token_end = token_start + TOKEN_BYTES;
    if mapping.get(token_start..token_end) != Some(token.as_bytes()) {
        return Err(RegionError::HeaderMismatch("token"));
    }
    let encoded_size: [u8; SIZE_BYTES] = mapping[token_end..HEADER_LEN]
        .try_into()
        .expect("header size field has a fixed width");
    if u64::from_le_bytes(encoded_size) != payload_size as u64 {
        return Err(RegionError::HeaderMismatch("size"));
    }
    Ok(())
}

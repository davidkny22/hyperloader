//! Validated names and mappings for cross-process arena regions.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io;
use std::ops::Range;

#[cfg(unix)]
#[path = "platform_unix.rs"]
mod platform;
#[cfg(windows)]
#[path = "platform_windows.rs"]
mod platform;

const NAME_PREFIX: &str = "/hl";
const BASE32: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";
const TOKEN_NAME_CHARS: usize = 18;
const SEQUENCE_NAME_CHARS: usize = 2;
const MAX_REGION_SEQUENCE: u16 = (1 << (SEQUENCE_NAME_CHARS * 5)) - 1;
const MAGIC: &[u8; 8] = b"HLARENA\0";
const TOKEN_BYTES: usize = 16;
const SIZE_BYTES: usize = size_of::<u64>();
const HEADER_LEN: usize = MAGIC.len() + TOKEN_BYTES + SIZE_BYTES;

/// A loader's full random identity, retained in every region header.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RegionToken([u8; TOKEN_BYTES]);

impl RegionToken {
    /// Construct a token from exact bytes, primarily for state restoration and tests.
    pub const fn from_bytes(bytes: [u8; TOKEN_BYTES]) -> Self {
        Self(bytes)
    }

    /// Generate a token from the operating system's secure random source.
    pub fn random() -> Result<Self, RegionError> {
        let mut bytes = [0_u8; TOKEN_BYTES];
        getrandom::fill(&mut bytes).map_err(|error| RegionError::Random(error.to_string()))?;
        Ok(Self(bytes))
    }

    /// Return the complete token used by region-header validation.
    pub const fn as_bytes(&self) -> &[u8; TOKEN_BYTES] {
        &self.0
    }
}

/// A portable named-region identifier with a fixed grammar.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RegionName(String);

impl RegionName {
    /// Derive a region name from the low token bits and bounded sequence number.
    pub fn new(token: RegionToken, sequence: u16) -> Result<Self, RegionError> {
        if sequence > MAX_REGION_SEQUENCE {
            return Err(RegionError::SequenceOutOfRange(sequence));
        }

        let low_bits = u128::from_le_bytes(token.0) & ((1_u128 << 90) - 1);
        let mut encoded = [0_u8; TOKEN_NAME_CHARS];
        encode_base32(low_bits, &mut encoded);
        let mut sequence_encoded = [0_u8; SEQUENCE_NAME_CHARS];
        encode_base32(sequence as u128, &mut sequence_encoded);

        let mut name =
            String::with_capacity(NAME_PREFIX.len() + TOKEN_NAME_CHARS + SEQUENCE_NAME_CHARS);
        name.push_str(NAME_PREFIX);
        // The alphabet is ASCII by construction.
        name.push_str(std::str::from_utf8(&encoded).expect("base32 alphabet must be UTF-8"));
        name.push_str(
            std::str::from_utf8(&sequence_encoded).expect("base32 alphabet must be UTF-8"),
        );
        Ok(Self(name))
    }

    /// Return the operating-system name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for RegionName {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

fn encode_base32(mut value: u128, output: &mut [u8]) {
    for character in output.iter_mut().rev() {
        *character = BASE32[(value & 31) as usize];
        value >>= 5;
    }
}

/// A typed failure from named-region construction, attachment, or validation.
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
    mapping: platform::Mapping,
}

impl NamedRegion {
    /// Create and initialize a region, failing if its name already exists.
    pub fn create(
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

#[cfg(test)]
mod tests {
    use super::{
        MAX_REGION_SEQUENCE, NamedRegion, RegionError, RegionName, RegionToken, TOKEN_BYTES,
    };
    use std::process::Command;

    fn unique_token() -> RegionToken {
        RegionToken::random().expect("operating-system random token")
    }

    #[test]
    fn name_uses_fixed_portable_grammar() {
        let token = RegionToken::from_bytes([0xff; 16]);
        let name = RegionName::new(token, MAX_REGION_SEQUENCE).expect("valid sequence");

        assert_eq!(name.as_str(), "/hl77777777777777777777");
        assert_eq!(name.as_str().len(), 23);
        assert!(
            name.as_str()
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'/')
        );
    }

    #[test]
    fn name_rejects_sequence_outside_two_base32_characters() {
        let error = RegionName::new(RegionToken::from_bytes([0; 16]), MAX_REGION_SEQUENCE + 1)
            .expect_err("out-of-range sequence must fail");
        assert!(matches!(error, RegionError::SequenceOutOfRange(_)));
    }

    #[test]
    fn random_tokens_are_not_reused() {
        let first = RegionToken::random().expect("first token");
        let second = RegionToken::random().expect("second token");
        assert_ne!(first, second);
    }

    #[test]
    fn region_shares_payload_and_rejects_exclusive_collision() {
        let token = unique_token();
        let mut owner = NamedRegion::create(token, 0, 64).expect("create owner");
        // SAFETY: the owner has exclusive access before any attachment exists.
        unsafe { owner.payload_mut()[..4].copy_from_slice(b"test") };

        let collision = NamedRegion::create(token, 0, 64)
            .err()
            .expect("exclusive create must reject a live name");
        assert!(matches!(collision, RegionError::AlreadyExists(_)));

        let attached = NamedRegion::attach(token, 0, 64).expect("attach by name");
        // SAFETY: the owner no longer writes and the attachment only reads.
        assert_eq!(unsafe { &attached.payload()[..4] }, b"test");
        owner.unlink().expect("unlink owner name");
    }

    #[test]
    fn attach_rejects_expected_size_mismatch() {
        let token = unique_token();
        let owner = NamedRegion::create(token, 1, 64).expect("create owner");
        let error = NamedRegion::attach(token, 1, 32)
            .err()
            .expect("size mismatch must fail");
        assert!(matches!(error, RegionError::HeaderMismatch("size")));
        owner.unlink().expect("unlink owner name");
    }

    #[test]
    fn attach_validates_token_bits_not_present_in_name() {
        let token = unique_token();
        let mut hostile_bytes = *token.as_bytes();
        hostile_bytes[TOKEN_BYTES - 1] ^= 0x80;
        let hostile_token = RegionToken::from_bytes(hostile_bytes);
        assert_eq!(
            RegionName::new(token, 2).expect("owner name"),
            RegionName::new(hostile_token, 2).expect("colliding hostile name")
        );

        let owner = NamedRegion::create(token, 2, 8).expect("create owner");
        let error = NamedRegion::attach(hostile_token, 2, 8)
            .err()
            .expect("full-token mismatch must fail");
        assert!(matches!(error, RegionError::HeaderMismatch("token")));
        owner.unlink().expect("unlink owner name");
    }

    #[test]
    fn attach_rejects_corrupted_magic() {
        let token = unique_token();
        let mut owner = NamedRegion::create(token, 3, 8).expect("create owner");
        owner.mapping.as_mut_slice()[0] ^= 0xff;

        let error = NamedRegion::attach(token, 3, 8)
            .err()
            .expect("corrupt magic must fail");
        assert!(matches!(error, RegionError::HeaderMismatch("magic")));
        owner.unlink().expect("unlink owner name");
    }

    #[test]
    fn region_attaches_from_an_independent_process() {
        let token = unique_token();
        let mut owner = NamedRegion::create(token, 4, 8).expect("create owner");
        // SAFETY: the owner has exclusive access before the child starts.
        unsafe { owner.payload_mut().copy_from_slice(b"process!") };
        let token_hex: String = token
            .as_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();

        let status = Command::new(std::env::current_exe().expect("current test executable"))
            .args([
                "--exact",
                "arena::named::tests::independent_process_attach_helper",
                "--ignored",
            ])
            .env("HYPERLOADER_TEST_REGION_TOKEN", token_hex)
            .status()
            .expect("launch attachment process");

        owner.unlink().expect("unlink owner name");
        assert!(status.success());
    }

    #[test]
    #[ignore = "launched by region_attaches_from_an_independent_process"]
    fn independent_process_attach_helper() {
        let encoded =
            std::env::var("HYPERLOADER_TEST_REGION_TOKEN").expect("attachment process token");
        assert_eq!(encoded.len(), TOKEN_BYTES * 2);
        let mut bytes = [0_u8; TOKEN_BYTES];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte =
                u8::from_str_radix(&encoded[index * 2..index * 2 + 2], 16).expect("hex token byte");
        }
        let region = NamedRegion::attach(RegionToken::from_bytes(bytes), 4, 8)
            .expect("attach in independent process");
        // SAFETY: the parent keeps the initialized payload read-only while the child runs.
        assert_eq!(unsafe { region.payload() }, b"process!");
    }

    #[test]
    fn zero_sized_regions_are_rejected_before_os_creation() {
        let error = NamedRegion::create(unique_token(), 2, 0)
            .err()
            .expect("zero-sized payload must fail");
        assert!(matches!(error, RegionError::InvalidSize(0)));
    }

    #[cfg(windows)]
    #[test]
    fn windows_name_lives_until_the_final_handle_closes() {
        let token = unique_token();
        {
            let owner = NamedRegion::create(token, 5, 8).expect("create owner");
            let attached = NamedRegion::attach(token, 5, 8).expect("attach second handle");
            owner.unlink().expect("request logical unlink");
            drop(owner);
            let third = NamedRegion::attach(token, 5, 8).expect("name remains while attached");
            drop(third);
            drop(attached);
        }

        let error = NamedRegion::attach(token, 5, 8)
            .err()
            .expect("name must disappear after the final handle closes");
        assert!(matches!(error, RegionError::NotFound(_)));
    }

    #[cfg(unix)]
    #[test]
    fn unlink_removes_name_without_invalidating_live_mapping() {
        let token = unique_token();
        let mut owner = NamedRegion::create(token, 3, 8).expect("create owner");
        // SAFETY: no attachment exists, so the owner has exclusive payload access.
        unsafe { owner.payload_mut()[0] = 41 };
        owner.unlink().expect("unlink name");

        let error = NamedRegion::attach(token, 3, 8)
            .err()
            .expect("unlinked name must not attach");
        assert!(matches!(error, RegionError::NotFound(_)));
        // SAFETY: the owner is the only remaining mapping and no writer is active.
        assert_eq!(unsafe { owner.payload()[0] }, 41);
    }

    #[cfg(unix)]
    #[test]
    fn created_region_has_owner_only_permissions() {
        use std::ffi::CString;

        let token = unique_token();
        let owner = NamedRegion::create(token, 5, 8).expect("create owner");
        let os_name = CString::new(owner.name().as_str()).expect("POSIX name");
        // SAFETY: `os_name` is a live NUL-terminated shared-memory name.
        let descriptor = unsafe { libc::shm_open(os_name.as_ptr(), libc::O_RDONLY, 0) };
        assert!(descriptor >= 0);
        // SAFETY: zeroed `stat` is valid output storage and `descriptor` is live.
        let mut metadata: libc::stat = unsafe { std::mem::zeroed() };
        // SAFETY: `metadata` points to writable storage and `descriptor` is live.
        assert_eq!(unsafe { libc::fstat(descriptor, &mut metadata) }, 0);
        // SAFETY: `descriptor` was returned open by `shm_open`.
        unsafe { libc::close(descriptor) };
        assert_eq!(metadata.st_mode & 0o777, 0o600);
        owner.unlink().expect("unlink owner name");
    }
}

//! Portable named-region identity grammar.

use super::RegionError;
use std::fmt::{Display, Formatter};

const NAME_PREFIX: &str = "/hl";
const BASE32: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";
const TOKEN_NAME_CHARS: usize = 18;
const SEQUENCE_NAME_CHARS: usize = 2;
pub(super) const MAX_REGION_SEQUENCE: u16 = (1 << (SEQUENCE_NAME_CHARS * 5)) - 1;
pub(super) const TOKEN_BYTES: usize = 16;

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

    pub(crate) fn to_hex(self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    pub(crate) fn from_hex(encoded: &str) -> Option<Self> {
        if encoded.len() != TOKEN_BYTES * 2 || !encoded.is_ascii() {
            return None;
        }
        let mut bytes = [0_u8; TOKEN_BYTES];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&encoded[index * 2..index * 2 + 2], 16).ok()?;
        }
        Some(Self(bytes))
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

    pub(crate) fn matches_token(&self, token: RegionToken) -> bool {
        let Some(sequence_text) = self
            .0
            .get(self.0.len().saturating_sub(SEQUENCE_NAME_CHARS)..)
        else {
            return false;
        };
        let Some(sequence) = decode_base32(sequence_text.as_bytes()) else {
            return false;
        };
        let Ok(expected) = Self::new(token, sequence as u16) else {
            return false;
        };
        expected == *self
    }

    pub(crate) fn from_registry(value: &str, token: RegionToken) -> Option<Self> {
        let candidate = Self(value.to_owned());
        candidate.matches_token(token).then_some(candidate)
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

fn decode_base32(input: &[u8]) -> Option<u128> {
    let mut value = 0_u128;
    for byte in input {
        let digit = BASE32.iter().position(|candidate| candidate == byte)? as u128;
        value = value.checked_mul(32)?.checked_add(digit)?;
    }
    Some(value)
}

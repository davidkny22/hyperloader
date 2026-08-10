//! Binary profile persistence behind an opaque cache-key path.

use super::{CostProfile, ProfileError};
use std::fs;
use std::path::Path;

const MAGIC: &[u8; 4] = b"HLPR";
const HEADER_BYTES: usize = 28;

pub(super) fn save(profile: &CostProfile, path: &Path) -> Result<(), ProfileError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(io_error("create profile cache directory"))?;
    }
    let mut bytes = Vec::with_capacity(HEADER_BYTES + profile.values.len() * 8);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&profile.position_count.to_le_bytes());
    bytes.extend_from_slice(&(profile.values.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&profile.alpha.to_bits().to_le_bytes());
    for value in &profile.values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    fs::write(path, bytes).map_err(io_error("write profile cache"))
}

pub(super) fn load(
    path: &Path,
    position_count: u64,
    max_bytes: u64,
    alpha: f64,
) -> Result<CostProfile, ProfileError> {
    let bytes = fs::read(path).map_err(io_error("read profile cache"))?;
    if bytes.len() < HEADER_BYTES || &bytes[..4] != MAGIC {
        return Err(ProfileError::new("profile cache header is invalid"));
    }
    let stored_positions = read_u64(&bytes, 4);
    let stored_entries = read_u64(&bytes, 12);
    let stored_alpha = f64::from_bits(read_u64(&bytes, 20));
    if stored_positions != position_count {
        return Err(ProfileError::new("profile cache position count changed"));
    }
    if stored_alpha.to_bits() != alpha.to_bits() {
        return Err(ProfileError::new("profile cache alpha changed"));
    }
    if stored_entries.saturating_mul(8) > max_bytes {
        return Err(ProfileError::new(
            "profile cache exceeds the current size budget",
        ));
    }
    let entries = usize::try_from(stored_entries)
        .map_err(|_| ProfileError::new("profile cache entry count exceeds this platform"))?;
    let expected_bytes = HEADER_BYTES
        .checked_add(
            entries
                .checked_mul(8)
                .ok_or_else(|| ProfileError::new("profile cache size overflowed"))?,
        )
        .ok_or_else(|| ProfileError::new("profile cache size overflowed"))?;
    if bytes.len() != expected_bytes || stored_entries > position_count {
        return Err(ProfileError::new("profile cache length is invalid"));
    }
    let mut values = Vec::with_capacity(entries);
    for offset in (HEADER_BYTES..expected_bytes).step_by(8) {
        let value = f64::from_bits(read_u64(&bytes, offset));
        if !value.is_nan() && (!value.is_finite() || value <= 0.0) {
            return Err(ProfileError::new("profile cache contains an invalid cost"));
        }
        values.push(value);
    }
    Ok(CostProfile {
        position_count,
        alpha,
        values,
    })
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("validated profile field"),
    )
}

fn io_error(operation: &'static str) -> impl FnOnce(std::io::Error) -> ProfileError {
    move |error| ProfileError::new(format!("{operation} failed: {error}"))
}

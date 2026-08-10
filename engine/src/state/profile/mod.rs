//! Bounded per-position execution-cost profiles.

mod store;

use crate::rng::splitmix64;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::Path;

const ENTRY_BYTES: u64 = size_of::<f64>() as u64;

/// A profile construction, observation, or persistence failure.
#[derive(Debug)]
pub struct ProfileError(String);

impl ProfileError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for ProfileError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ProfileError {}

/// Aggregate statistics used by adaptive frontier sizing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProfileStatistics {
    /// Mean of the populated position or bucket estimates.
    pub mean_ns: f64,
    /// Nearest-rank 99.9th percentile of populated estimates.
    pub p999_ns: f64,
    /// Number of populated profile entries.
    pub populated: usize,
}

/// An EMA cost table that degrades to deterministic position-hash buckets.
#[derive(Clone, Debug)]
pub struct CostProfile {
    position_count: u64,
    alpha: f64,
    values: Vec<f64>,
}

impl CostProfile {
    /// Create a profile whose value payload never exceeds `max_bytes`.
    pub fn new(position_count: u64, max_bytes: u64, alpha: f64) -> Result<Self, ProfileError> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) || alpha == 0.0 {
            return Err(ProfileError::new("profile alpha must be in (0, 1]"));
        }
        let budget_entries = max_bytes / ENTRY_BYTES;
        let entry_count = position_count.min(budget_entries);
        let entry_count = usize::try_from(entry_count)
            .map_err(|_| ProfileError::new("profile entry count exceeds this platform"))?;
        Ok(Self {
            position_count,
            alpha,
            values: vec![f64::NAN; entry_count],
        })
    }

    /// Record one positive measured wall cost in nanoseconds.
    pub fn observe(&mut self, position: u64, cost_ns: u64) -> Result<(), ProfileError> {
        if position >= self.position_count {
            return Err(ProfileError::new("profile position is outside its domain"));
        }
        if cost_ns == 0 {
            return Err(ProfileError::new("profile cost must be positive"));
        }
        let Some(index) = self.entry_index(position) else {
            return Ok(());
        };
        let observation = cost_ns as f64;
        let current = self.values[index];
        self.values[index] = if current.is_nan() {
            observation
        } else {
            self.alpha
                .mul_add(observation, (1.0 - self.alpha) * current)
        };
        Ok(())
    }

    /// Return the current estimate for one position when any observation exists.
    pub fn estimate(&self, position: u64) -> Result<Option<f64>, ProfileError> {
        if position >= self.position_count {
            return Err(ProfileError::new("profile position is outside its domain"));
        }
        let Some(index) = self.entry_index(position) else {
            return Ok(None);
        };
        Ok((!self.values[index].is_nan()).then_some(self.values[index]))
    }

    /// Summarize populated estimates for frontier sizing.
    pub fn statistics(&self) -> Option<ProfileStatistics> {
        let mut populated: Vec<f64> = self
            .values
            .iter()
            .copied()
            .filter(|value| !value.is_nan())
            .collect();
        if populated.is_empty() {
            return None;
        }
        let mean_ns = populated.iter().sum::<f64>() / populated.len() as f64;
        populated.sort_by(f64::total_cmp);
        let rank = ((populated.len() as f64 * 0.999).ceil() as usize)
            .saturating_sub(1)
            .min(populated.len() - 1);
        Some(ProfileStatistics {
            mean_ns,
            p999_ns: populated[rank],
            populated: populated.len(),
        })
    }

    /// Persist the bounded profile at an opaque cache-key path.
    pub fn save(&self, path: &Path) -> Result<(), ProfileError> {
        store::save(self, path)
    }

    /// Load a profile and validate its domain, budget, and EMA factor.
    pub fn load(
        path: &Path,
        position_count: u64,
        max_bytes: u64,
        alpha: f64,
    ) -> Result<Self, ProfileError> {
        store::load(path, position_count, max_bytes, alpha)
    }

    /// Return whether estimates share hashed buckets because of the size clamp.
    pub fn is_degraded(&self) -> bool {
        (self.values.len() as u64) < self.position_count
    }

    /// Return the exact value-payload size.
    pub fn payload_bytes(&self) -> u64 {
        self.values.len() as u64 * ENTRY_BYTES
    }

    fn entry_index(&self, position: u64) -> Option<usize> {
        if self.values.is_empty() {
            return None;
        }
        if !self.is_degraded() {
            return Some(position as usize);
        }
        Some((splitmix64(position) % self.values.len() as u64) as usize)
    }
}

#[cfg(test)]
mod tests;

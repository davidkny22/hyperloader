//! Validated names and mappings for cross-process arena regions.

mod identity;
mod region;

#[cfg(unix)]
#[path = "platform_unix.rs"]
mod platform;
#[cfg(windows)]
#[path = "platform_windows.rs"]
mod platform;

pub use identity::{RegionName, RegionToken};
pub(crate) use region::unlink_registered;
pub use region::{NamedRegion, RegionError};

#[cfg(test)]
use identity::{MAX_REGION_SEQUENCE, TOKEN_BYTES};

#[cfg(test)]
mod tests;

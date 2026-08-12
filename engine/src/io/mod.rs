//! Platform file reads and their refuge selection.

mod contract;
mod plan;
mod refuge;

#[cfg(windows)]
mod platform_windows;

pub use contract::{BackendKind, BackendPreference, IoError, ReadCompletion};
pub use plan::PlatformBackend;

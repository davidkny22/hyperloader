//! Platform cache directory resolution for the region registry.

use std::path::PathBuf;

#[cfg(windows)]
pub(super) fn user_cache_directory() -> Option<PathBuf> {
    configured_cache_directory().or_else(|| std::env::var_os("LOCALAPPDATA").map(PathBuf::from))
}

#[cfg(target_os = "macos")]
pub(super) fn user_cache_directory() -> Option<PathBuf> {
    configured_cache_directory().or_else(|| {
        std::env::var_os("HOME").map(|home| PathBuf::from(home).join("Library").join("Caches"))
    })
}

#[cfg(all(unix, not(target_os = "macos")))]
pub(super) fn user_cache_directory() -> Option<PathBuf> {
    configured_cache_directory().or_else(|| {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
    })
}

fn configured_cache_directory() -> Option<PathBuf> {
    std::env::var_os("HYPERLOADER_CACHE_HOME").map(PathBuf::from)
}

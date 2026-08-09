//! Atomic POSIX registry replacement.

use super::RegistryError;
use std::fs::{self, File};
use std::path::Path;

pub(super) fn atomic_replace(source: &Path, destination: &Path) -> Result<(), RegistryError> {
    fs::rename(source, destination)
        .map_err(|error| RegistryError::io("replace region registry", error))
}

pub(super) fn sync_parent(parent: &Path) -> Result<(), RegistryError> {
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| RegistryError::io("synchronize region registry directory", error))
}

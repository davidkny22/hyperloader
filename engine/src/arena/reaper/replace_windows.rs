//! Atomic Windows registry replacement.

use super::RegistryError;
use std::io;
use std::path::Path;
use windows_sys::Win32::Storage::FileSystem::{
    MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
};

pub(super) fn atomic_replace(source: &Path, destination: &Path) -> Result<(), RegistryError> {
    let source = wide_path(source);
    let destination = wide_path(destination);
    // SAFETY: both paths are live NUL-terminated UTF-16 strings for the duration of the call.
    if unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    } == 0
    {
        return Err(RegistryError::io(
            "replace region registry",
            io::Error::last_os_error(),
        ));
    }
    Ok(())
}

pub(super) fn sync_parent(_parent: &Path) -> Result<(), RegistryError> {
    // MOVEFILE_WRITE_THROUGH waits for the replacement to reach persistent storage.
    Ok(())
}

fn wide_path(path: &Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;

    path.as_os_str().encode_wide().chain(Some(0)).collect()
}

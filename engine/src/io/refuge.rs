//! Positioned-read refuge compiled on every supported platform.

use super::{IoError, ReadCompletion};
use std::fs::File;
use std::path::Path;

pub(super) fn read_into(
    path: &Path,
    offset: u64,
    destination: &mut [u8],
) -> Result<ReadCompletion, IoError> {
    let file = File::open(path).map_err(|error| IoError::os("open input file", error))?;
    let mut total = 0_usize;
    while total < destination.len() {
        let current_offset = offset
            .checked_add(total as u64)
            .ok_or(IoError::OffsetOverflow)?;
        let read = positioned_read(&file, &mut destination[total..], current_offset)
            .map_err(|error| IoError::os("read input range", error))?;
        if read == 0 {
            break;
        }
        total += read;
    }
    Ok(ReadCompletion::new(total))
}

#[cfg(unix)]
fn positioned_read(file: &File, destination: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;

    file.read_at(destination, offset)
}

#[cfg(windows)]
fn positioned_read(file: &File, destination: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;

    file.seek_read(destination, offset)
}

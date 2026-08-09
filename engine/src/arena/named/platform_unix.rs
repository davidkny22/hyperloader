//! POSIX named shared-memory mapping implementation.

use super::{RegionError, RegionName};
use std::ffi::CString;
use std::io;
use std::ptr::NonNull;
use std::slice;

pub(super) struct Mapping {
    address: NonNull<u8>,
    length: usize,
}

// SAFETY: an mmap view is process-wide and may move between threads. Mutable access to the
// mapped bytes still requires exclusive access to `Mapping`.
unsafe impl Send for Mapping {}

impl Mapping {
    pub(super) fn create(name: &RegionName, length: usize) -> Result<Self, RegionError> {
        let os_name = os_name(name)?;
        // SAFETY: `os_name` is a live NUL-terminated string and the flags and mode are valid.
        let descriptor = unsafe {
            libc::shm_open(
                os_name.as_ptr(),
                libc::O_CREAT | libc::O_EXCL | libc::O_RDWR,
                (libc::S_IRUSR | libc::S_IWUSR) as libc::c_uint,
            )
        };
        if descriptor < 0 {
            let source = io::Error::last_os_error();
            return if source.raw_os_error() == Some(libc::EEXIST) {
                Err(RegionError::AlreadyExists(name.clone()))
            } else {
                Err(RegionError::os("shm_open create", source))
            };
        }

        let file_length =
            libc::off_t::try_from(length).map_err(|_| RegionError::InvalidSize(length));
        let result = match file_length {
            Ok(file_length) => {
                // SAFETY: `descriptor` is open for writing and this is its sole sizing call.
                if unsafe { libc::ftruncate(descriptor, file_length) } != 0 {
                    Err(RegionError::os(
                        "ftruncate shared region",
                        io::Error::last_os_error(),
                    ))
                } else {
                    map_descriptor(descriptor, length)
                }
            }
            Err(error) => Err(error),
        };

        // SAFETY: `descriptor` was returned open by `shm_open` and is no longer needed after mmap.
        unsafe { libc::close(descriptor) };
        if result.is_err() {
            // SAFETY: `os_name` remains live; cleanup is best effort after failed creation.
            unsafe { libc::shm_unlink(os_name.as_ptr()) };
        }
        result
    }

    pub(super) fn open(name: &RegionName, length: usize) -> Result<Self, RegionError> {
        let os_name = os_name(name)?;
        // SAFETY: `os_name` is a live NUL-terminated string and the access flag is valid.
        let descriptor = unsafe { libc::shm_open(os_name.as_ptr(), libc::O_RDWR, 0) };
        if descriptor < 0 {
            let source = io::Error::last_os_error();
            return if source.raw_os_error() == Some(libc::ENOENT) {
                Err(RegionError::NotFound(name.clone()))
            } else {
                Err(RegionError::os("shm_open attach", source))
            };
        }

        // SAFETY: zeroed `stat` is valid output storage and `descriptor` is live.
        let mut metadata: libc::stat = unsafe { std::mem::zeroed() };
        // SAFETY: `metadata` points to writable storage and `descriptor` is live.
        let stat_result = unsafe { libc::fstat(descriptor, &mut metadata) };
        let result = if stat_result != 0 {
            Err(RegionError::os(
                "fstat shared region",
                io::Error::last_os_error(),
            ))
        } else if usize::try_from(metadata.st_size).ok() != Some(length) {
            Err(RegionError::HeaderMismatch("size"))
        } else {
            map_descriptor(descriptor, length)
        };
        // SAFETY: `descriptor` was returned open by `shm_open` and is no longer needed.
        unsafe { libc::close(descriptor) };
        result
    }

    pub(super) fn as_slice(&self) -> &[u8] {
        // SAFETY: the mapping owns `length` readable bytes for its entire lifetime.
        unsafe { slice::from_raw_parts(self.address.as_ptr(), self.length) }
    }

    pub(super) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `&mut self` provides exclusive Rust access to the local mapped view.
        unsafe { slice::from_raw_parts_mut(self.address.as_ptr(), self.length) }
    }
}

fn map_descriptor(descriptor: libc::c_int, length: usize) -> Result<Mapping, RegionError> {
    // SAFETY: `descriptor` names a region of `length` bytes and the mapping flags are valid.
    let address = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            length,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            descriptor,
            0,
        )
    };
    if address == libc::MAP_FAILED {
        return Err(RegionError::os(
            "mmap shared region",
            io::Error::last_os_error(),
        ));
    }
    let address = NonNull::new(address.cast::<u8>()).expect("mmap success is non-null");
    Ok(Mapping { address, length })
}

pub(super) fn unlink(name: &RegionName) -> Result<(), RegionError> {
    let os_name = os_name(name)?;
    // SAFETY: `os_name` is a live NUL-terminated POSIX shared-memory name.
    if unsafe { libc::shm_unlink(os_name.as_ptr()) } != 0 {
        let source = io::Error::last_os_error();
        return if source.raw_os_error() == Some(libc::ENOENT) {
            Err(RegionError::NotFound(name.clone()))
        } else {
            Err(RegionError::os("shm_unlink", source))
        };
    }
    Ok(())
}

fn os_name(name: &RegionName) -> Result<CString, RegionError> {
    CString::new(name.as_str()).map_err(|_| {
        RegionError::os(
            "encode shared region name",
            io::Error::new(io::ErrorKind::InvalidInput, "region name contains NUL"),
        )
    })
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: this exact address and length came from one successful `mmap` call.
        unsafe { libc::munmap(self.address.as_ptr().cast(), self.length) };
    }
}

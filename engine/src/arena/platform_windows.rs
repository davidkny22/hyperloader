//! Windows named pagefile-mapping implementation.

use super::{RegionError, RegionName};
use std::io;
use std::ptr::NonNull;
use std::slice;
use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_ALREADY_EXISTS, ERROR_FILE_NOT_FOUND, GetLastError, HANDLE,
    INVALID_HANDLE_VALUE, LocalFree,
};
use windows_sys::Win32::Security::Authorization::{
    ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
};
use windows_sys::Win32::Security::{PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_ALL_ACCESS, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
    OpenFileMappingW, PAGE_READWRITE, UnmapViewOfFile,
};

pub(super) struct Mapping {
    handle: HANDLE,
    view: MEMORY_MAPPED_VIEW_ADDRESS,
    address: NonNull<u8>,
    length: usize,
}

impl Mapping {
    pub(super) fn create(name: &RegionName, length: usize) -> Result<Self, RegionError> {
        let os_name = os_name(name);
        let length = u64::try_from(length).map_err(|_| RegionError::InvalidSize(length))?;
        let security = OwnerSecurity::new()?;
        // SAFETY: the name is NUL-terminated, `security` remains live for the call, and the
        // high and low words describe one pagefile-backed mapping size.
        let handle = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                security.attributes(),
                PAGE_READWRITE,
                (length >> 32) as u32,
                length as u32,
                os_name.as_ptr(),
            )
        };
        if handle.is_null() {
            return Err(RegionError::os(
                "CreateFileMappingW",
                io::Error::last_os_error(),
            ));
        }
        // SAFETY: `GetLastError` is read immediately after the successful creation call.
        if unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
            // SAFETY: `handle` is a live mapping handle returned above.
            unsafe { CloseHandle(handle) };
            return Err(RegionError::AlreadyExists(name.clone()));
        }
        map_handle(handle, length as usize)
    }

    pub(super) fn open(name: &RegionName, length: usize) -> Result<Self, RegionError> {
        let os_name = os_name(name);
        // SAFETY: the name is NUL-terminated and the requested mapping access is valid.
        let handle = unsafe { OpenFileMappingW(FILE_MAP_ALL_ACCESS, 0, os_name.as_ptr()) };
        if handle.is_null() {
            let source = io::Error::last_os_error();
            return if source.raw_os_error() == Some(ERROR_FILE_NOT_FOUND as i32) {
                Err(RegionError::NotFound(name.clone()))
            } else {
                Err(RegionError::os("OpenFileMappingW", source))
            };
        }
        map_handle(handle, length)
    }

    pub(super) fn as_slice(&self) -> &[u8] {
        // SAFETY: the mapped view owns `length` readable bytes for its entire lifetime.
        unsafe { slice::from_raw_parts(self.address.as_ptr(), self.length) }
    }

    pub(super) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `&mut self` provides exclusive Rust access to the local mapped view.
        unsafe { slice::from_raw_parts_mut(self.address.as_ptr(), self.length) }
    }
}

struct OwnerSecurity {
    descriptor: PSECURITY_DESCRIPTOR,
    attributes: SECURITY_ATTRIBUTES,
}

impl OwnerSecurity {
    fn new() -> Result<Self, RegionError> {
        let sddl: Vec<u16> = "D:P(A;;GA;;;OW)".encode_utf16().chain(Some(0)).collect();
        let mut descriptor = std::ptr::null_mut();
        // SAFETY: `sddl` is NUL-terminated and `descriptor` points to writable output storage.
        if unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl.as_ptr(),
                SDDL_REVISION_1,
                &mut descriptor,
                std::ptr::null_mut(),
            )
        } == 0
        {
            return Err(RegionError::os(
                "build owner-only security descriptor",
                io::Error::last_os_error(),
            ));
        }
        let attributes = SECURITY_ATTRIBUTES {
            nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: descriptor,
            bInheritHandle: 0,
        };
        Ok(Self {
            descriptor,
            attributes,
        })
    }

    fn attributes(&self) -> *const SECURITY_ATTRIBUTES {
        &self.attributes
    }
}

impl Drop for OwnerSecurity {
    fn drop(&mut self) {
        // SAFETY: the descriptor was allocated by the conversion function and is freed once.
        unsafe { LocalFree(self.descriptor) };
    }
}

fn map_handle(handle: HANDLE, length: usize) -> Result<Mapping, RegionError> {
    // SAFETY: `handle` is a live file-mapping handle and the requested length is its size.
    let view = unsafe { MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, length) };
    let Some(address) = NonNull::new(view.Value.cast::<u8>()) else {
        let source = io::Error::last_os_error();
        // SAFETY: `handle` is live and no view was created.
        unsafe { CloseHandle(handle) };
        return Err(RegionError::os("MapViewOfFile", source));
    };
    Ok(Mapping {
        handle,
        view,
        address,
        length,
    })
}

pub(super) fn unlink(_name: &RegionName) -> Result<(), RegionError> {
    // Windows mapping names disappear automatically after the final handle closes.
    Ok(())
}

fn os_name(name: &RegionName) -> Vec<u16> {
    name.as_str().encode_utf16().chain(Some(0)).collect()
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: this view and handle came from one successful mapping operation.
        unsafe {
            UnmapViewOfFile(self.view);
            CloseHandle(self.handle);
        }
    }
}

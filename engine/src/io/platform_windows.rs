//! Windows overlapped reads completed through I/O completion ports.

use super::{IoError, ReadCompletion};
use std::io;
use std::os::windows::ffi::OsStrExt;
use std::path::Path;
use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_HANDLE_EOF, ERROR_IO_PENDING, GENERIC_READ, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, FILE_ATTRIBUTE_NORMAL, FILE_FLAG_OVERLAPPED, FILE_SHARE_DELETE, FILE_SHARE_READ,
    FILE_SHARE_WRITE, OPEN_EXISTING, ReadFile,
};
use windows_sys::Win32::System::IO::{
    CreateIoCompletionPort, GetQueuedCompletionStatus, OVERLAPPED, OVERLAPPED_0, OVERLAPPED_0_0,
};
use windows_sys::Win32::System::Threading::INFINITE;

const COMPLETION_KEY: usize = 1;

pub(super) struct IocpBackend;

impl IocpBackend {
    pub(super) fn read_into(
        &self,
        path: &Path,
        offset: u64,
        destination: &mut [u8],
    ) -> Result<ReadCompletion, IoError> {
        if destination.is_empty() {
            return Ok(ReadCompletion::new(0));
        }
        let length = u32::try_from(destination.len())
            .map_err(|_| IoError::InvalidLength(destination.len()))?;
        let file = OwnedHandle::open_overlapped(path)?;
        let port = OwnedHandle::completion_port()?;
        port.associate(file.handle())?;

        let mut overlapped = OVERLAPPED {
            Anonymous: OVERLAPPED_0 {
                Anonymous: OVERLAPPED_0_0 {
                    Offset: offset as u32,
                    OffsetHigh: (offset >> 32) as u32,
                },
            },
            ..OVERLAPPED::default()
        };
        // SAFETY: the file and destination remain live until their matching completion packet is
        // removed, and `overlapped` has a stable stack address for the same interval.
        let submitted = unsafe {
            ReadFile(
                file.handle(),
                destination.as_mut_ptr(),
                length,
                std::ptr::null_mut(),
                &mut overlapped,
            )
        };
        if submitted == 0 {
            let source = io::Error::last_os_error();
            match source.raw_os_error() {
                Some(code) if code == ERROR_IO_PENDING as i32 => {}
                Some(code) if code == ERROR_HANDLE_EOF as i32 => {
                    return Ok(ReadCompletion::new(0));
                }
                _ => return Err(IoError::os("submit overlapped file read", source)),
            }
        }

        let mut transferred = 0_u32;
        let mut key = 0_usize;
        let mut completed = std::ptr::null_mut();
        // SAFETY: the port is live and all output pointers reference writable storage. The wait
        // ends only after the submitted request has produced a completion packet.
        let status = unsafe {
            GetQueuedCompletionStatus(
                port.handle(),
                &mut transferred,
                &mut key,
                &mut completed,
                INFINITE,
            )
        };
        if status == 0 {
            let source = io::Error::last_os_error();
            if source.raw_os_error() == Some(ERROR_HANDLE_EOF as i32) {
                return Ok(ReadCompletion::new(0));
            }
            return Err(IoError::os("receive file-read completion", source));
        }
        if key != COMPLETION_KEY || completed != &mut overlapped {
            return Err(IoError::CompletionMismatch);
        }
        Ok(ReadCompletion::new(transferred as usize))
    }
}

struct OwnedHandle(HANDLE);

impl OwnedHandle {
    fn open_overlapped(path: &Path) -> Result<Self, IoError> {
        let path: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        // SAFETY: the path is NUL-terminated, no security structure is provided, and the returned
        // handle is either invalid or owned by this call.
        let handle = unsafe {
            CreateFileW(
                path.as_ptr(),
                GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                std::ptr::null(),
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
                std::ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return Err(IoError::os(
                "open overlapped input file",
                io::Error::last_os_error(),
            ));
        }
        Ok(Self(handle))
    }

    fn completion_port() -> Result<Self, IoError> {
        // SAFETY: INVALID_HANDLE_VALUE requests a new completion port and supplies no existing
        // port. The returned handle is either null or owned by this call.
        let handle =
            unsafe { CreateIoCompletionPort(INVALID_HANDLE_VALUE, std::ptr::null_mut(), 0, 1) };
        if handle.is_null() {
            return Err(IoError::os(
                "create I/O completion port",
                io::Error::last_os_error(),
            ));
        }
        Ok(Self(handle))
    }

    fn associate(&self, file: HANDLE) -> Result<(), IoError> {
        // SAFETY: both handles are live, and this request gives the file a stable completion key.
        let result = unsafe { CreateIoCompletionPort(file, self.0, COMPLETION_KEY, 0) };
        if result.is_null() {
            return Err(IoError::os(
                "associate input file with I/O completion port",
                io::Error::last_os_error(),
            ));
        }
        if result != self.0 {
            return Err(IoError::CompletionMismatch);
        }
        Ok(())
    }

    const fn handle(&self) -> HANDLE {
        self.0
    }
}

impl Drop for OwnedHandle {
    fn drop(&mut self) {
        // SAFETY: this object owns one live handle and closes it exactly once.
        unsafe { CloseHandle(self.0) };
    }
}

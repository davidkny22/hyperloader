//! Windows overlapped reads completed through I/O completion ports.

use super::{IoError, ReadCompletion};
use std::collections::HashMap;
use std::io;
use std::os::windows::ffi::OsStrExt;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
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

pub(super) struct IocpBackend {
    port: Arc<CompletionPort>,
}

impl IocpBackend {
    pub(super) fn new() -> Result<Self, IoError> {
        Ok(Self {
            port: Arc::new(CompletionPort::new()?),
        })
    }

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
        let key = self.port.next_key();
        self.port.associate(file.handle(), key)?;
        let (sender, receiver) = mpsc::channel();

        let mut overlapped = OVERLAPPED {
            Anonymous: OVERLAPPED_0 {
                Anonymous: OVERLAPPED_0_0 {
                    Offset: offset as u32,
                    OffsetHigh: (offset >> 32) as u32,
                },
            },
            ..OVERLAPPED::default()
        };
        self.port.register(key, &mut overlapped, sender)?;
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
                    self.port.cancel_registration(key);
                    return Ok(ReadCompletion::new(0));
                }
                _ => {
                    self.port.cancel_registration(key);
                    return Err(IoError::os("submit overlapped file read", source));
                }
            }
        }
        self.port.wait(receiver).map(ReadCompletion::new)
    }
}

struct CompletionPort {
    handle: OwnedHandle,
    next_key: AtomicUsize,
    waiters: Mutex<HashMap<usize, Waiter>>,
    pump: Mutex<()>,
}

// SAFETY: the completion-port handle is process-global kernel state whose concurrent use is
// defined by IOCP. Rust access to the request registry and receive pump is mutex-protected.
unsafe impl Send for CompletionPort {}
// SAFETY: the same handle invariant applies to shared references, and request buffers remain
// owned by their submitting callers until the matching packet is dispatched.
unsafe impl Sync for CompletionPort {}

struct Waiter {
    overlapped: usize,
    sender: mpsc::Sender<Result<usize, IoError>>,
}

impl CompletionPort {
    fn new() -> Result<Self, IoError> {
        Ok(Self {
            handle: OwnedHandle::completion_port()?,
            next_key: AtomicUsize::new(1),
            waiters: Mutex::new(HashMap::new()),
            pump: Mutex::new(()),
        })
    }

    fn next_key(&self) -> usize {
        loop {
            let key = self.next_key.fetch_add(1, Ordering::Relaxed);
            if key != 0 {
                return key;
            }
        }
    }

    fn associate(&self, file: HANDLE, key: usize) -> Result<(), IoError> {
        // SAFETY: both handles are live, and this request gives the file a unique completion key.
        let result = unsafe { CreateIoCompletionPort(file, self.handle.handle(), key, 0) };
        if result.is_null() {
            return Err(IoError::os(
                "associate input file with I/O completion port",
                io::Error::last_os_error(),
            ));
        }
        if result != self.handle.handle() {
            return Err(IoError::CompletionMismatch);
        }
        Ok(())
    }

    fn register(
        &self,
        key: usize,
        overlapped: *mut OVERLAPPED,
        sender: mpsc::Sender<Result<usize, IoError>>,
    ) -> Result<(), IoError> {
        let replaced = self
            .waiters
            .lock()
            .map_err(|_| IoError::CompletionMismatch)?
            .insert(
                key,
                Waiter {
                    overlapped: overlapped as usize,
                    sender,
                },
            );
        if replaced.is_some() {
            return Err(IoError::CompletionMismatch);
        }
        Ok(())
    }

    fn cancel_registration(&self, key: usize) {
        if let Ok(mut waiters) = self.waiters.lock() {
            waiters.remove(&key);
        }
    }

    fn wait(&self, receiver: mpsc::Receiver<Result<usize, IoError>>) -> Result<usize, IoError> {
        loop {
            match receiver.try_recv() {
                Ok(result) => return result,
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err(IoError::CompletionMismatch);
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
            let _pump = self.pump.lock().map_err(|_| IoError::CompletionMismatch)?;
            match receiver.try_recv() {
                Ok(result) => return result,
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err(IoError::CompletionMismatch);
                }
                Err(mpsc::TryRecvError::Empty) => self.pump_one()?,
            }
        }
    }

    fn pump_one(&self) -> Result<(), IoError> {
        let mut transferred = 0_u32;
        let mut key = 0_usize;
        let mut completed = std::ptr::null_mut();
        // SAFETY: the port is live and all output pointers reference writable storage. At least
        // one registered request remains pending while the single receive pump waits.
        let status = unsafe {
            GetQueuedCompletionStatus(
                self.handle.handle(),
                &mut transferred,
                &mut key,
                &mut completed,
                INFINITE,
            )
        };
        let source = (status == 0).then(io::Error::last_os_error);
        let waiter = self
            .waiters
            .lock()
            .map_err(|_| IoError::CompletionMismatch)?
            .remove(&key)
            .ok_or(IoError::CompletionMismatch)?;
        let result = if waiter.overlapped != completed as usize {
            Err(IoError::CompletionMismatch)
        } else if let Some(source) = source {
            if source.raw_os_error() == Some(ERROR_HANDLE_EOF as i32) {
                Ok(0)
            } else {
                Err(IoError::os("receive file-read completion", source))
            }
        } else {
            Ok(transferred as usize)
        };
        waiter
            .sender
            .send(result)
            .map_err(|_| IoError::CompletionMismatch)
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

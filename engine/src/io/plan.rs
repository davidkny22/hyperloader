//! Plan-time backend selection and the common read surface.

use super::{BackendKind, BackendPreference, IoError, ReadCompletion, refuge};
use std::path::Path;

#[cfg(windows)]
use super::platform_windows::IocpBackend;

/// One concrete backend selected once for a loader plan.
pub struct PlatformBackend {
    inner: Backend,
}

enum Backend {
    #[cfg(windows)]
    Iocp(IocpBackend),
    Pread,
}

impl PlatformBackend {
    /// Resolve a configuration preference without runtime fallback after selection.
    pub fn select(preference: BackendPreference) -> Result<Self, IoError> {
        select_platform(preference)
    }

    /// Return the concrete backend chosen for the plan.
    pub const fn kind(&self) -> BackendKind {
        match self.inner {
            #[cfg(windows)]
            Backend::Iocp(_) => BackendKind::Iocp,
            Backend::Pread => BackendKind::Pread,
        }
    }

    /// Read a file range into its final destination and return its completion record.
    pub fn read_into(
        &self,
        path: &Path,
        offset: u64,
        destination: &mut [u8],
    ) -> Result<ReadCompletion, IoError> {
        match &self.inner {
            #[cfg(windows)]
            Backend::Iocp(backend) => backend.read_into(path, offset, destination),
            Backend::Pread => refuge::read_into(path, offset, destination),
        }
    }

    /// Read a file range into an owned result, truncating it at end of file.
    pub fn read_range(&self, path: &Path, offset: u64, length: usize) -> Result<Vec<u8>, IoError> {
        let mut output = vec![0_u8; length];
        let completion = self.read_into(path, offset, &mut output)?;
        output.truncate(completion.bytes_read());
        Ok(output)
    }
}

#[cfg(windows)]
fn select_platform(preference: BackendPreference) -> Result<PlatformBackend, IoError> {
    match preference {
        BackendPreference::Auto | BackendPreference::Iocp => Ok(PlatformBackend {
            inner: Backend::Iocp(IocpBackend::new()?),
        }),
        BackendPreference::Pread => Ok(PlatformBackend {
            inner: Backend::Pread,
        }),
        BackendPreference::Uring => Err(IoError::Unavailable {
            requested: "uring",
            platform: std::env::consts::OS,
        }),
    }
}

#[cfg(not(windows))]
fn select_platform(preference: BackendPreference) -> Result<PlatformBackend, IoError> {
    match preference {
        BackendPreference::Auto | BackendPreference::Pread => Ok(PlatformBackend {
            inner: Backend::Pread,
        }),
        BackendPreference::Uring => Err(IoError::Unavailable {
            requested: "uring",
            platform: std::env::consts::OS,
        }),
        BackendPreference::Iocp => Err(IoError::Unavailable {
            requested: "iocp",
            platform: std::env::consts::OS,
        }),
    }
}

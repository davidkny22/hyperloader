//! Locked JSONL persistence and atomic compaction.

use super::cache::user_cache_directory;
use super::format::{parse_snapshot, validate_entry};
use super::model::{RegistryEntry, RegistryError, RegistrySnapshot};
use super::sweep::SweepReport;
use crate::arena::reaper::{ProcessObserver, SystemProcessObserver};
use crate::arena::{NamedRegion, RegionToken};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

#[cfg(unix)]
#[path = "replace_unix.rs"]
mod replace;
#[cfg(windows)]
#[path = "replace_windows.rs"]
mod replace;

/// A registry whose stable sibling lock protects append and atomic compaction.
#[derive(Clone, Debug)]
pub struct RegionRegistry {
    path: PathBuf,
    lock_path: PathBuf,
}

impl RegionRegistry {
    /// Bind a registry to an explicit path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let lock_path = path.with_extension("lock");
        Self { path, lock_path }
    }

    /// Resolve the current user's cache registry without creating it.
    pub fn for_current_user() -> Result<Self, RegistryError> {
        let cache = user_cache_directory().ok_or_else(|| {
            RegistryError::io(
                "resolve user cache directory",
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "user cache directory is unavailable",
                ),
            )
        })?;
        Ok(Self::new(cache.join("hyperloader").join("regions.jsonl")))
    }

    /// Resolve the user registry and run the guarded construction-time sweep.
    pub fn prepare_current_user() -> Result<(Self, SweepReport), RegistryError> {
        let registry = Self::for_current_user()?;
        let report = registry.sweep()?;
        Ok((registry, report))
    }

    /// Return the JSONL data path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append and synchronize one complete ownership record under the stable lock.
    pub fn append(&self, entry: &RegistryEntry) -> Result<(), RegistryError> {
        validate_entry(entry)?;
        self.ensure_parent()?;
        let lock = self.open_lock()?;
        lock.lock()
            .map_err(|error| RegistryError::io("lock region registry", error))?;
        let result = self.append_locked(entry);
        drop(lock);
        result
    }

    /// Append one region-create record using the current guarded process identity.
    pub fn register(&self, region: &NamedRegion) -> Result<RegistryEntry, RegistryError> {
        self.register_with(region, &SystemProcessObserver)
    }

    /// Exclusively create a region and persist its ownership before returning it.
    pub fn create_region(
        &self,
        token: RegionToken,
        sequence: u16,
        payload_size: usize,
    ) -> Result<(NamedRegion, RegistryEntry), RegistryError> {
        let region =
            NamedRegion::create(token, sequence, payload_size).map_err(RegistryError::Region)?;
        match self.register(&region) {
            Ok(entry) => Ok((region, entry)),
            Err(error) => {
                let _ = region.unlink();
                Err(error)
            }
        }
    }

    fn register_with<O: ProcessObserver>(
        &self,
        region: &NamedRegion,
        observer: &O,
    ) -> Result<RegistryEntry, RegistryError> {
        let identity = observer
            .current_identity()
            .map_err(RegistryError::Identity)?;
        let entry = RegistryEntry::new(
            region,
            std::process::id(),
            identity.boot_id,
            identity.proc_start,
        );
        self.append(&entry)?;
        Ok(entry)
    }

    /// Read a consistent registry snapshot.
    pub fn snapshot(&self) -> Result<RegistrySnapshot, RegistryError> {
        self.ensure_parent()?;
        let lock = self.open_lock()?;
        lock.lock_shared()
            .map_err(|error| RegistryError::io("lock region registry for reading", error))?;
        let result = self.read_locked();
        drop(lock);
        result
    }

    /// Filter valid records and atomically compact while holding the append lock.
    ///
    /// Corrupt or ambiguous input is returned unchanged and suppresses replacement.
    pub fn retain<F>(&self, mut keep: F) -> Result<RegistrySnapshot, RegistryError>
    where
        F: FnMut(&RegistryEntry) -> bool,
    {
        self.ensure_parent()?;
        let lock = self.open_lock()?;
        lock.lock()
            .map_err(|error| RegistryError::io("lock region registry", error))?;
        let mut snapshot = self.read_locked()?;
        if snapshot.issues.is_empty() {
            snapshot.entries.retain(|entry| keep(entry));
            self.replace_locked(&snapshot.entries)?;
        }
        drop(lock);
        Ok(snapshot)
    }

    pub(super) fn ensure_parent(&self) -> Result<(), RegistryError> {
        let parent = self.path.parent().ok_or_else(|| {
            RegistryError::io(
                "resolve region registry directory",
                io::Error::new(io::ErrorKind::InvalidInput, "registry path has no parent"),
            )
        })?;
        fs::create_dir_all(parent)
            .map_err(|error| RegistryError::io("create region registry directory", error))
    }

    fn open_lock(&self) -> Result<File, RegistryError> {
        let file = private_options()
            .read(true)
            .write(true)
            .create(true)
            .open(&self.lock_path)
            .map_err(|error| RegistryError::io("open region registry lock", error))?;
        harden_permissions(&file)?;
        Ok(file)
    }

    fn append_locked(&self, entry: &RegistryEntry) -> Result<(), RegistryError> {
        let mut line = serde_json::to_vec(entry)?;
        line.push(b'\n');
        let mut file = private_options()
            .read(true)
            .append(true)
            .create(true)
            .open(&self.path)
            .map_err(|error| RegistryError::io("open region registry", error))?;
        harden_permissions(&file)?;
        file.write_all(&line)
            .map_err(|error| RegistryError::io("append region registry", error))?;
        file.sync_all()
            .map_err(|error| RegistryError::io("synchronize region registry", error))
    }

    fn read_locked(&self) -> Result<RegistrySnapshot, RegistryError> {
        let mut bytes = Vec::new();
        match File::open(&self.path) {
            Ok(mut file) => {
                file.read_to_end(&mut bytes)
                    .map_err(|error| RegistryError::io("read region registry", error))?;
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(RegistrySnapshot::default());
            }
            Err(error) => return Err(RegistryError::io("open region registry", error)),
        }
        Ok(parse_snapshot(&bytes))
    }

    fn replace_locked(&self, entries: &[RegistryEntry]) -> Result<(), RegistryError> {
        let parent = self.path.parent().expect("validated registry parent");
        let mut random = [0_u8; 8];
        getrandom::fill(&mut random).map_err(|error| {
            RegistryError::io(
                "generate registry temporary name",
                io::Error::other(error.to_string()),
            )
        })?;
        let suffix: String = random.iter().map(|byte| format!("{byte:02x}")).collect();
        let temporary = parent.join(format!(".regions-{suffix}.tmp"));
        let result = (|| {
            let mut file = private_options()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .map_err(|error| RegistryError::io("create registry replacement", error))?;
            for entry in entries {
                validate_entry(entry)?;
                serde_json::to_writer(&mut file, entry)?;
                file.write_all(b"\n")
                    .map_err(|error| RegistryError::io("write registry replacement", error))?;
            }
            file.sync_all()
                .map_err(|error| RegistryError::io("synchronize registry replacement", error))?;
            drop(file);
            replace::atomic_replace(&temporary, &self.path)?;
            replace::sync_parent(parent)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }
}

#[cfg(unix)]
pub(super) fn private_options() -> OpenOptions {
    use std::os::unix::fs::OpenOptionsExt;

    let mut options = OpenOptions::new();
    options.mode(0o600);
    options
}

#[cfg(unix)]
fn harden_permissions(file: &File) -> Result<(), RegistryError> {
    use std::os::unix::fs::PermissionsExt;

    file.set_permissions(fs::Permissions::from_mode(0o600))
        .map_err(|error| RegistryError::io("restrict region registry permissions", error))
}

#[cfg(windows)]
pub(super) fn private_options() -> OpenOptions {
    OpenOptions::new()
}

#[cfg(windows)]
fn harden_permissions(_file: &File) -> Result<(), RegistryError> {
    Ok(())
}

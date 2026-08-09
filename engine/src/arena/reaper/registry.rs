//! Locked JSONL persistence for region ownership records.

use crate::arena::{NamedRegion, RegionName, RegionToken};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

#[cfg(unix)]
#[path = "replace_unix.rs"]
mod replace;
#[cfg(windows)]
#[path = "replace_windows.rs"]
mod replace;

/// One durable ownership record written when a named region is created.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RegistryEntry {
    /// Portable shared-memory name.
    pub name: String,
    /// Complete loader token in lowercase hexadecimal.
    pub token: String,
    /// Creator process identifier.
    pub pid: u32,
    /// Boot identity observed by the creator.
    pub boot_id: String,
    /// Creator process start time in platform-normalized units.
    pub proc_start: u64,
}

impl RegistryEntry {
    /// Build an ownership record from a validated region and process identity.
    pub fn new(
        region: &NamedRegion,
        pid: u32,
        boot_id: impl Into<String>,
        proc_start: u64,
    ) -> Self {
        Self {
            name: region.name().as_str().to_owned(),
            token: region.token().to_hex(),
            pid,
            boot_id: boot_id.into(),
            proc_start,
        }
    }

    pub(crate) fn validated_name(&self) -> Option<RegionName> {
        let token = RegionToken::from_hex(&self.token)?;
        RegionName::from_registry(&self.name, token)
    }
}

/// A recoverable problem found while reading registry evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistryIssue {
    /// One-based line number, or zero for a file-level issue.
    pub line: usize,
    /// Stable issue description for audit logging.
    pub message: String,
}

/// A locked registry read with valid entries and preserved audit issues.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RegistrySnapshot {
    /// Valid, unique ownership records.
    pub entries: Vec<RegistryEntry>,
    /// Corruption or ambiguity that suppresses destructive cleanup.
    pub issues: Vec<RegistryIssue>,
}

/// A typed registry persistence failure.
#[derive(Debug)]
pub enum RegistryError {
    /// A filesystem operation failed.
    Io {
        /// Operation that failed.
        operation: &'static str,
        /// Original filesystem error.
        source: io::Error,
    },
    /// A record could not be serialized.
    Json(serde_json::Error),
    /// A caller attempted to append a malformed or inconsistent record.
    InvalidEntry,
}

impl RegistryError {
    fn io(operation: &'static str, source: io::Error) -> Self {
        Self::Io { operation, source }
    }
}

impl Display for RegistryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { operation, source } => write!(formatter, "{operation} failed: {source}"),
            Self::Json(source) => write!(formatter, "registry serialization failed: {source}"),
            Self::InvalidEntry => formatter.write_str("registry entry is malformed"),
        }
    }
}

impl Error for RegistryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json(source) => Some(source),
            Self::InvalidEntry => None,
        }
    }
}

impl From<serde_json::Error> for RegistryError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

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

    fn ensure_parent(&self) -> Result<(), RegistryError> {
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
        private_options()
            .read(true)
            .write(true)
            .create(true)
            .open(&self.lock_path)
            .map_err(|error| RegistryError::io("open region registry lock", error))
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

fn validate_entry(entry: &RegistryEntry) -> Result<(), RegistryError> {
    if entry.pid == 0
        || entry.boot_id.is_empty()
        || entry.proc_start == 0
        || entry.validated_name().is_none()
    {
        return Err(RegistryError::InvalidEntry);
    }
    Ok(())
}

fn parse_snapshot(bytes: &[u8]) -> RegistrySnapshot {
    let mut snapshot = RegistrySnapshot::default();
    let mut names = HashSet::new();
    let mut offset = 0;
    let mut line_number = 1;
    while offset < bytes.len() {
        let Some(relative_end) = bytes[offset..].iter().position(|byte| *byte == b'\n') else {
            snapshot.issues.push(RegistryIssue {
                line: line_number,
                message: "registry ends with an incomplete line".to_owned(),
            });
            break;
        };
        let end = offset + relative_end;
        let line = &bytes[offset..end];
        match serde_json::from_slice::<RegistryEntry>(line) {
            Ok(entry) if validate_entry(&entry).is_ok() && names.insert(entry.name.clone()) => {
                snapshot.entries.push(entry);
            }
            Ok(_) => snapshot.issues.push(RegistryIssue {
                line: line_number,
                message: "registry entry is malformed or duplicated".to_owned(),
            }),
            Err(_) => snapshot.issues.push(RegistryIssue {
                line: line_number,
                message: "registry line is not valid JSON".to_owned(),
            }),
        }
        offset = end + 1;
        line_number += 1;
    }
    snapshot
}

#[cfg(unix)]
fn private_options() -> OpenOptions {
    use std::os::unix::fs::OpenOptionsExt;

    let mut options = OpenOptions::new();
    options.mode(0o600);
    options
}

#[cfg(windows)]
fn private_options() -> OpenOptions {
    OpenOptions::new()
}

#[cfg(test)]
mod tests {
    use super::{RegionRegistry, RegistryEntry};
    use crate::arena::{NamedRegion, RegionToken};
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::sync::{Arc, Barrier};
    use std::thread;

    fn temporary_registry(label: &str) -> RegionRegistry {
        let mut random = [0_u8; 8];
        getrandom::fill(&mut random).expect("temporary registry randomness");
        let suffix: String = random.iter().map(|byte| format!("{byte:02x}")).collect();
        RegionRegistry::new(std::env::temp_dir().join(format!("hl-{label}-{suffix}/regions.jsonl")))
    }

    fn entry(sequence: u16, pid: u32) -> RegistryEntry {
        let token = RegionToken::random().expect("region token");
        let region = NamedRegion::create(token, sequence, 8).expect("temporary named region");
        let entry = RegistryEntry::new(&region, pid, "boot", u64::from(pid) + 1);
        region.unlink().expect("unlink temporary region");
        entry
    }

    #[test]
    fn append_round_trips_complete_lines() {
        let registry = temporary_registry("roundtrip");
        let first = entry(0, 10);
        let second = entry(1, 11);
        registry.append(&first).expect("append first");
        registry.append(&second).expect("append second");

        let snapshot = registry.snapshot().expect("read snapshot");
        assert_eq!(snapshot.entries, [first, second]);
        assert!(snapshot.issues.is_empty());
    }

    #[test]
    fn incomplete_tail_is_preserved_as_ambiguity() {
        let registry = temporary_registry("truncated");
        let valid = entry(0, 12);
        registry.append(&valid).expect("append valid entry");
        let mut file = OpenOptions::new()
            .append(true)
            .open(registry.path())
            .expect("open registry tail");
        file.write_all(b"{\"name\":")
            .expect("write incomplete tail");
        file.sync_all().expect("synchronize incomplete tail");

        let snapshot = registry.snapshot().expect("read damaged registry");
        assert_eq!(snapshot.entries.as_slice(), std::slice::from_ref(&valid));
        assert_eq!(snapshot.issues.len(), 1);
        let retained = registry.retain(|_| false).expect("conservative retain");
        assert_eq!(retained.entries, [valid]);
        assert_eq!(retained.issues.len(), 1);
    }

    #[test]
    fn concurrent_append_and_compaction_lose_no_entries() {
        const WRITERS: usize = 4;
        const ENTRIES_PER_WRITER: usize = 8;
        let registry = Arc::new(temporary_registry("concurrent"));
        let barrier = Arc::new(Barrier::new(WRITERS + 1));
        let mut writers = Vec::new();
        for writer in 0..WRITERS {
            let registry = Arc::clone(&registry);
            let barrier = Arc::clone(&barrier);
            writers.push(thread::spawn(move || {
                barrier.wait();
                for index in 0..ENTRIES_PER_WRITER {
                    registry
                        .append(&entry(
                            (writer * ENTRIES_PER_WRITER + index) as u16,
                            (100 + writer * ENTRIES_PER_WRITER + index) as u32,
                        ))
                        .expect("concurrent append");
                }
            }));
        }
        barrier.wait();
        for _ in 0..ENTRIES_PER_WRITER {
            registry.retain(|_| true).expect("concurrent compaction");
        }
        for writer in writers {
            writer.join().expect("writer thread");
        }

        let snapshot = registry.snapshot().expect("final snapshot");
        assert!(snapshot.issues.is_empty());
        assert_eq!(snapshot.entries.len(), WRITERS * ENTRIES_PER_WRITER);
    }
}

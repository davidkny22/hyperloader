//! Locked JSONL persistence for region ownership records.

use super::process::{ProcessObservation, ProcessObserver, SystemProcessObserver};
use crate::arena::named::unlink_registered;
use crate::arena::{NamedRegion, RegionError, RegionName, RegionToken};
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
    /// Current process identity could not be established safely.
    Identity(String),
    /// Named-region creation failed before registration.
    Region(RegionError),
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
            Self::Identity(message) => write!(formatter, "process identity failed: {message}"),
            Self::Region(source) => write!(formatter, "named region creation failed: {source}"),
        }
    }
}

impl Error for RegistryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json(source) => Some(source),
            Self::InvalidEntry => None,
            Self::Identity(_) => None,
            Self::Region(source) => Some(source),
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

/// The reason a sweep retained or removed one ownership record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SweepOutcome {
    /// The recorded process still owns the identity.
    Live,
    /// The recorded process is absent.
    Dead,
    /// The PID now belongs to another process or boot.
    Reused,
    /// The observation or unlink operation could not prove safety.
    Ambiguous,
}

/// One audit action emitted by a guarded sweep.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SweepAction {
    /// Region name from the validated registry record.
    pub name: String,
    /// Guard decision.
    pub outcome: SweepOutcome,
    /// Concise observation detail.
    pub detail: String,
}

/// Complete result of one guarded sweep and compaction attempt.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SweepReport {
    /// Records retained as live.
    pub live: usize,
    /// Records removed after death or reuse was proven.
    pub removed: usize,
    /// Records retained because safety could not be proven.
    pub ambiguous: usize,
    /// Per-record audit actions.
    pub actions: Vec<SweepAction>,
    /// Registry corruption that suppressed all destructive work.
    pub registry_issues: Vec<RegistryIssue>,
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

    /// Sweep stale ownership records using the system process observer.
    pub fn sweep(&self) -> Result<SweepReport, RegistryError> {
        self.sweep_with(&SystemProcessObserver)
    }

    fn sweep_with<O: ProcessObserver>(&self, observer: &O) -> Result<SweepReport, RegistryError> {
        let mut report = SweepReport::default();
        let snapshot = self.retain(|entry| {
            let (keep, outcome, detail) = match observer.observe(entry.pid) {
                ProcessObservation::Missing => match unlink_entry(entry) {
                    Ok(()) => (false, SweepOutcome::Dead, "process is absent".to_owned()),
                    Err(detail) => (true, SweepOutcome::Ambiguous, detail),
                },
                ProcessObservation::Live(identity)
                    if identity.boot_id == entry.boot_id
                        && identity.proc_start == entry.proc_start =>
                {
                    (
                        true,
                        SweepOutcome::Live,
                        "process identity matches".to_owned(),
                    )
                }
                ProcessObservation::Live(_) => match unlink_entry(entry) {
                    Ok(()) => (
                        false,
                        SweepOutcome::Reused,
                        "process identity differs".to_owned(),
                    ),
                    Err(detail) => (true, SweepOutcome::Ambiguous, detail),
                },
                ProcessObservation::Ambiguous(detail) => (true, SweepOutcome::Ambiguous, detail),
            };
            match outcome {
                SweepOutcome::Live => report.live += 1,
                SweepOutcome::Dead | SweepOutcome::Reused => report.removed += 1,
                SweepOutcome::Ambiguous => report.ambiguous += 1,
            }
            report.actions.push(SweepAction {
                name: entry.name.clone(),
                outcome,
                detail,
            });
            keep
        })?;
        if !snapshot.issues.is_empty() {
            report.ambiguous = snapshot.entries.len();
            report.registry_issues = snapshot.issues;
        }
        Ok(report)
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

fn unlink_entry(entry: &RegistryEntry) -> Result<(), String> {
    let name = entry
        .validated_name()
        .ok_or_else(|| "registry identity is malformed".to_owned())?;
    match unlink_registered(&name) {
        Ok(()) | Err(RegionError::NotFound(_)) => Ok(()),
        Err(error) => Err(format!("region unlink remained ambiguous: {error}")),
    }
}

#[cfg(windows)]
fn user_cache_directory() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
}

#[cfg(target_os = "macos")]
fn user_cache_directory() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join("Library").join("Caches"))
}

#[cfg(all(unix, not(target_os = "macos")))]
fn user_cache_directory() -> Option<PathBuf> {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
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

#[cfg(unix)]
fn harden_permissions(file: &File) -> Result<(), RegistryError> {
    use std::os::unix::fs::PermissionsExt;

    file.set_permissions(fs::Permissions::from_mode(0o600))
        .map_err(|error| RegistryError::io("restrict region registry permissions", error))
}

#[cfg(windows)]
fn private_options() -> OpenOptions {
    OpenOptions::new()
}

#[cfg(windows)]
fn harden_permissions(_file: &File) -> Result<(), RegistryError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{RegionRegistry, RegistryEntry, SweepOutcome, private_options};
    use crate::arena::reaper::{ProcessIdentity, ProcessObservation, ProcessObserver};
    use crate::arena::{NamedRegion, RegionToken};
    use std::collections::HashMap;
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

    fn owned_entry(sequence: u16, pid: u32) -> (NamedRegion, RegistryEntry) {
        let token = RegionToken::random().expect("region token");
        let region = NamedRegion::create(token, sequence, 8).expect("temporary named region");
        let entry = RegistryEntry::new(&region, pid, "boot", u64::from(pid) + 1);
        (region, entry)
    }

    struct FakeObserver {
        current: ProcessIdentity,
        observations: HashMap<u32, ProcessObservation>,
    }

    impl ProcessObserver for FakeObserver {
        fn current_identity(&self) -> Result<ProcessIdentity, String> {
            Ok(self.current.clone())
        }

        fn observe(&self, pid: u32) -> ProcessObservation {
            self.observations
                .get(&pid)
                .cloned()
                .unwrap_or_else(|| ProcessObservation::Ambiguous("unmapped test PID".to_owned()))
        }
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
    fn hostile_name_suppresses_destructive_compaction() {
        let registry = temporary_registry("hostile-name");
        registry.ensure_parent().expect("create registry directory");
        let hostile = RegistryEntry {
            name: "/unrelated-object".to_owned(),
            token: "00".repeat(16),
            pid: 13,
            boot_id: "boot".to_owned(),
            proc_start: 14,
        };
        let mut file = private_options()
            .write(true)
            .create_new(true)
            .open(registry.path())
            .expect("create hostile registry");
        serde_json::to_writer(&mut file, &hostile).expect("write hostile record");
        file.write_all(b"\n").expect("complete hostile line");
        file.sync_all().expect("synchronize hostile registry");

        let snapshot = registry.snapshot().expect("read hostile registry");
        assert!(snapshot.entries.is_empty());
        assert_eq!(snapshot.issues.len(), 1);
        let report = registry.sweep().expect("conservative hostile sweep");
        assert_eq!(report.registry_issues.len(), 1);
        assert!(registry.path().exists());
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

    #[test]
    fn system_identity_registration_is_retained_while_live() {
        let registry = temporary_registry("system-live");
        let token = RegionToken::random().expect("region token");
        let (region, registered) = registry
            .create_region(token, 40, 8)
            .expect("create registered region");

        let report = registry.sweep().expect("guarded system sweep");
        assert_eq!(report.live, 1);
        assert_eq!(report.removed, 0);
        assert_eq!(report.ambiguous, 0);
        assert_eq!(report.actions[0].outcome, SweepOutcome::Live);
        assert_eq!(
            registry.snapshot().expect("live snapshot").entries,
            [registered]
        );
        region.unlink().expect("unlink live region");
        registry.retain(|_| false).expect("clear test registry");
    }

    #[test]
    fn sweep_removes_only_proven_dead_or_reused_owners() {
        let registry = temporary_registry("guarded-sweep");
        let (live_region, live) = owned_entry(41, 201);
        let (dead_region, dead) = owned_entry(42, 202);
        let (reused_region, reused) = owned_entry(43, 203);
        let (ambiguous_region, ambiguous) = owned_entry(44, 204);
        for entry in [&live, &dead, &reused, &ambiguous] {
            registry.append(entry).expect("append sweep record");
        }
        let observer = FakeObserver {
            current: ProcessIdentity {
                boot_id: "boot".to_owned(),
                proc_start: 1,
            },
            observations: HashMap::from([
                (
                    live.pid,
                    ProcessObservation::Live(ProcessIdentity {
                        boot_id: live.boot_id.clone(),
                        proc_start: live.proc_start,
                    }),
                ),
                (dead.pid, ProcessObservation::Missing),
                (
                    reused.pid,
                    ProcessObservation::Live(ProcessIdentity {
                        boot_id: "different-boot".to_owned(),
                        proc_start: reused.proc_start,
                    }),
                ),
                (
                    ambiguous.pid,
                    ProcessObservation::Ambiguous("permission denied".to_owned()),
                ),
            ]),
        };

        let report = registry.sweep_with(&observer).expect("guarded fake sweep");
        assert_eq!(report.live, 1);
        assert_eq!(report.removed, 2);
        assert_eq!(report.ambiguous, 1);
        assert_eq!(
            report
                .actions
                .iter()
                .map(|action| action.outcome.clone())
                .collect::<Vec<_>>(),
            [
                SweepOutcome::Live,
                SweepOutcome::Dead,
                SweepOutcome::Reused,
                SweepOutcome::Ambiguous,
            ]
        );
        let snapshot = registry.snapshot().expect("compacted sweep registry");
        assert_eq!(snapshot.entries, [live, ambiguous]);

        live_region.unlink().expect("unlink retained live region");
        ambiguous_region
            .unlink()
            .expect("unlink retained ambiguous region");
        drop(dead_region);
        drop(reused_region);
    }

    #[cfg(unix)]
    #[test]
    fn registry_files_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let registry = temporary_registry("permissions");
        registry
            .append(&entry(45, 205))
            .expect("append registry entry");
        let data_mode = std::fs::metadata(registry.path())
            .expect("registry metadata")
            .permissions()
            .mode()
            & 0o777;
        let lock_mode = std::fs::metadata(registry.path().with_extension("lock"))
            .expect("lock metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(data_mode, 0o600);
        assert_eq!(lock_mode, 0o600);
    }
}

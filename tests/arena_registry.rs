use _hyperloader::arena::{RegionName, RegionRegistry, RegionToken, RegistryEntry, SweepOutcome};
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
    RegistryEntry {
        name: RegionName::new(token, sequence)
            .expect("region name")
            .as_str()
            .to_owned(),
        token: token
            .as_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
        pid,
        boot_id: "boot".to_owned(),
        proc_start: u64::from(pid) + 1,
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

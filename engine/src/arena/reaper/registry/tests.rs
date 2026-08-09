use super::{RegionRegistry, RegistryEntry, SweepOutcome, private_options};
use crate::arena::reaper::{ProcessIdentity, ProcessObservation, ProcessObserver};
use crate::arena::{NamedRegion, RegionToken};
use std::collections::HashMap;
use std::io::Write;

fn temporary_registry(label: &str) -> RegionRegistry {
    let mut random = [0_u8; 8];
    getrandom::fill(&mut random).expect("temporary registry randomness");
    let suffix: String = random.iter().map(|byte| format!("{byte:02x}")).collect();
    RegionRegistry::new(std::env::temp_dir().join(format!("hl-{label}-{suffix}/regions.jsonl")))
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

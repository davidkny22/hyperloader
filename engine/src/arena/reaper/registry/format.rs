//! Validation and decoding for crash-atomic registry records.

use super::model::{RegistryEntry, RegistryError, RegistryIssue, RegistrySnapshot};
use std::collections::HashSet;

pub(super) fn validate_entry(entry: &RegistryEntry) -> Result<(), RegistryError> {
    if entry.pid == 0
        || entry.boot_id.is_empty()
        || entry.proc_start == 0
        || entry.validated_name().is_none()
    {
        return Err(RegistryError::InvalidEntry);
    }
    Ok(())
}

pub(super) fn parse_snapshot(bytes: &[u8]) -> RegistrySnapshot {
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

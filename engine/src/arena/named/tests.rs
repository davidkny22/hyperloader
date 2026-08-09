use super::{MAX_REGION_SEQUENCE, NamedRegion, RegionError, RegionName, RegionToken, TOKEN_BYTES};
use std::process::Command;

fn unique_token() -> RegionToken {
    RegionToken::random().expect("operating-system random token")
}

#[test]
fn name_uses_fixed_portable_grammar() {
    let token = RegionToken::from_bytes([0xff; 16]);
    let name = RegionName::new(token, MAX_REGION_SEQUENCE).expect("valid sequence");

    assert_eq!(name.as_str(), "/hl77777777777777777777");
    assert_eq!(name.as_str().len(), 23);
    assert!(
        name.as_str()
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'/')
    );
}

#[test]
fn name_rejects_sequence_outside_two_base32_characters() {
    let error = RegionName::new(RegionToken::from_bytes([0; 16]), MAX_REGION_SEQUENCE + 1)
        .expect_err("out-of-range sequence must fail");
    assert!(matches!(error, RegionError::SequenceOutOfRange(_)));
}

#[test]
fn region_shares_payload_and_rejects_exclusive_collision() {
    let token = unique_token();
    let mut owner = NamedRegion::create(token, 0, 64).expect("create owner");
    // SAFETY: the owner has exclusive access before any attachment exists.
    unsafe { owner.payload_mut()[..4].copy_from_slice(b"test") };

    let collision = NamedRegion::create(token, 0, 64)
        .err()
        .expect("exclusive create must reject a live name");
    assert!(matches!(collision, RegionError::AlreadyExists(_)));

    let attached = NamedRegion::attach(token, 0, 64).expect("attach by name");
    // SAFETY: the owner no longer writes and the attachment only reads.
    assert_eq!(unsafe { &attached.payload()[..4] }, b"test");
    owner.unlink().expect("unlink owner name");
}

#[test]
fn attach_rejects_expected_size_mismatch() {
    let token = unique_token();
    let owner = NamedRegion::create(token, 1, 64).expect("create owner");
    let error = NamedRegion::attach(token, 1, 32)
        .err()
        .expect("size mismatch must fail");
    assert!(matches!(error, RegionError::HeaderMismatch("size")));
    owner.unlink().expect("unlink owner name");
}

#[test]
fn attach_validates_token_bits_not_present_in_name() {
    let token = unique_token();
    let mut hostile_bytes = *token.as_bytes();
    hostile_bytes[TOKEN_BYTES - 1] ^= 0x80;
    let hostile_token = RegionToken::from_bytes(hostile_bytes);
    assert_eq!(
        RegionName::new(token, 2).expect("owner name"),
        RegionName::new(hostile_token, 2).expect("colliding hostile name")
    );

    let owner = NamedRegion::create(token, 2, 8).expect("create owner");
    let error = NamedRegion::attach(hostile_token, 2, 8)
        .err()
        .expect("full-token mismatch must fail");
    assert!(matches!(error, RegionError::HeaderMismatch("token")));
    owner.unlink().expect("unlink owner name");
}

#[test]
fn attach_rejects_corrupted_magic() {
    let token = unique_token();
    let mut owner = NamedRegion::create(token, 3, 8).expect("create owner");
    owner.mapping.as_mut_slice()[0] ^= 0xff;

    let error = NamedRegion::attach(token, 3, 8)
        .err()
        .expect("corrupt magic must fail");
    assert!(matches!(error, RegionError::HeaderMismatch("magic")));
    owner.unlink().expect("unlink owner name");
}

#[test]
fn region_attaches_from_an_independent_process() {
    let token = unique_token();
    let mut owner = NamedRegion::create(token, 4, 8).expect("create owner");
    // SAFETY: the owner has exclusive access before the child starts.
    unsafe { owner.payload_mut().copy_from_slice(b"process!") };
    let token_hex: String = token
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();

    let status = Command::new(std::env::current_exe().expect("current test executable"))
        .args([
            "--exact",
            "arena::named::tests::independent_process_attach_helper",
            "--ignored",
        ])
        .env("HYPERLOADER_TEST_REGION_TOKEN", token_hex)
        .status()
        .expect("launch attachment process");

    owner.unlink().expect("unlink owner name");
    assert!(status.success());
}

#[test]
#[ignore = "launched by region_attaches_from_an_independent_process"]
fn independent_process_attach_helper() {
    let encoded = std::env::var("HYPERLOADER_TEST_REGION_TOKEN").expect("attachment process token");
    assert_eq!(encoded.len(), TOKEN_BYTES * 2);
    let mut bytes = [0_u8; TOKEN_BYTES];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&encoded[index * 2..index * 2 + 2], 16).expect("hex token byte");
    }
    let region = NamedRegion::attach(RegionToken::from_bytes(bytes), 4, 8)
        .expect("attach in independent process");
    // SAFETY: the parent keeps the initialized payload read-only while the child runs.
    assert_eq!(unsafe { region.payload() }, b"process!");
}

#[test]
fn zero_sized_regions_are_rejected_before_os_creation() {
    let error = NamedRegion::create(unique_token(), 2, 0)
        .err()
        .expect("zero-sized payload must fail");
    assert!(matches!(error, RegionError::InvalidSize(0)));
}

#[cfg(windows)]
#[test]
fn windows_name_lives_until_the_final_handle_closes() {
    let token = unique_token();
    {
        let owner = NamedRegion::create(token, 5, 8).expect("create owner");
        let attached = NamedRegion::attach(token, 5, 8).expect("attach second handle");
        owner.unlink().expect("request logical unlink");
        drop(owner);
        let third = NamedRegion::attach(token, 5, 8).expect("name remains while attached");
        drop(third);
        drop(attached);
    }

    let error = NamedRegion::attach(token, 5, 8)
        .err()
        .expect("name must disappear after the final handle closes");
    assert!(matches!(error, RegionError::NotFound(_)));
}

#[cfg(unix)]
#[test]
fn unlink_removes_name_without_invalidating_live_mapping() {
    let token = unique_token();
    let mut owner = NamedRegion::create(token, 3, 8).expect("create owner");
    // SAFETY: no attachment exists, so the owner has exclusive payload access.
    unsafe { owner.payload_mut()[0] = 41 };
    owner.unlink().expect("unlink name");

    let error = NamedRegion::attach(token, 3, 8)
        .err()
        .expect("unlinked name must not attach");
    assert!(matches!(error, RegionError::NotFound(_)));
    // SAFETY: the owner is the only remaining mapping and no writer is active.
    assert_eq!(unsafe { owner.payload()[0] }, 41);
}

#[cfg(unix)]
#[test]
fn created_region_has_owner_only_permissions() {
    use std::ffi::CString;

    let token = unique_token();
    let owner = NamedRegion::create(token, 5, 8).expect("create owner");
    let os_name = CString::new(owner.name().as_str()).expect("POSIX name");
    // SAFETY: `os_name` is a live NUL-terminated shared-memory name.
    let descriptor = unsafe { libc::shm_open(os_name.as_ptr(), libc::O_RDONLY, 0) };
    assert!(descriptor >= 0);
    // SAFETY: zeroed `stat` is valid output storage and `descriptor` is live.
    let mut metadata: libc::stat = unsafe { std::mem::zeroed() };
    // SAFETY: `metadata` points to writable storage and `descriptor` is live.
    assert_eq!(unsafe { libc::fstat(descriptor, &mut metadata) }, 0);
    // SAFETY: `descriptor` was returned open by `shm_open`.
    unsafe { libc::close(descriptor) };
    assert_eq!(metadata.st_mode & 0o777, 0o600);
    owner.unlink().expect("unlink owner name");
}

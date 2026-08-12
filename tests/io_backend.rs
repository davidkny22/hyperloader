use _hyperloader::io::{BackendKind, BackendPreference, PlatformBackend};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    path: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let sequence = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "hyperloader-io-{}-{sequence}.bin",
            std::process::id()
        ));
        fs::write(&path, b"0123456789abcdef").expect("write input fixture");
        Self { path }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[test]
fn pread_refuge_reads_exact_ranges_and_stops_at_eof() {
    let fixture = Fixture::new();
    let backend = PlatformBackend::select(BackendPreference::Pread).expect("pread backend");
    assert_eq!(backend.kind(), BackendKind::Pread);
    assert_eq!(
        backend
            .read_range(&fixture.path, 3, 6)
            .expect("middle range"),
        b"345678"
    );
    assert_eq!(
        backend
            .read_range(&fixture.path, 14, 8)
            .expect("short final range"),
        b"ef"
    );
    assert!(
        backend
            .read_range(&fixture.path, 30, 4)
            .expect("range after end")
            .is_empty()
    );
}

#[test]
fn unknown_backend_is_rejected_before_platform_selection() {
    let error = "other"
        .parse::<BackendPreference>()
        .expect_err("unknown backend must fail");
    assert!(error.to_string().contains("unknown I/O backend"));
}

#[cfg(windows)]
#[test]
fn windows_auto_selects_iocp_and_receives_matching_completions() {
    let fixture = Fixture::new();
    let backend = PlatformBackend::select(BackendPreference::Auto).expect("automatic backend");
    assert_eq!(backend.kind(), BackendKind::Iocp);
    assert_eq!(
        backend.read_range(&fixture.path, 5, 7).expect("IOCP range"),
        b"56789ab"
    );
    assert_eq!(
        backend
            .read_range(&fixture.path, 15, 3)
            .expect("IOCP final range"),
        b"f"
    );
}

#[cfg(windows)]
#[test]
fn windows_iocp_dispatches_concurrent_completions_to_their_callers() {
    let fixture = Fixture::new();
    let backend =
        Arc::new(PlatformBackend::select(BackendPreference::Iocp).expect("shared IOCP backend"));
    let expected = b"0123456789abcdef";
    let mut threads = Vec::new();
    for offset in 0..expected.len() {
        let backend = Arc::clone(&backend);
        let path = fixture.path.clone();
        threads.push(std::thread::spawn(move || {
            backend
                .read_range(&path, offset as u64, 1)
                .expect("concurrent range")
        }));
    }
    for (offset, thread) in threads.into_iter().enumerate() {
        assert_eq!(
            thread.join().expect("reader thread"),
            expected[offset..=offset]
        );
    }
}

#[cfg(not(windows))]
#[test]
fn iocp_is_rejected_off_windows() {
    let error = PlatformBackend::select(BackendPreference::Iocp)
        .err()
        .expect("IOCP must be unavailable");
    assert!(error.to_string().contains("unavailable"));
}

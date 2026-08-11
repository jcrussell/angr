//! Demotion of symbolic file content to concrete: what `demote_symbolic_content`
//! must clear (fds, registry, cwd-drifted paths), the demoted-path tracking
//! that makes a re-demote a no-op, and the write choke point that refuses a
//! write rather than silently dropping symbolic content.
//!
//! Carved out of the former monolithic `state_tests.rs` (angr-c7xno.69).

use super::super::*;
use super::helpers::sym_file_bytes;

#[test]
fn test_demote_symbolic_content_clears_all_fds_and_registry() {
    let mut fs = FileSystem::default();
    fs.register_file_content("/tmp/d", sym_file_bytes(4, "dem"));
    // Three handles on the same file: absolute, relative spelling, dup.
    let fd1 = fs.open("/tmp/d".to_string(), FdFlags::ReadWrite).expect("fd space is not exhausted in tests");
    let fd2 = fs.open("tmp/d".to_string(), FdFlags::ReadOnly).expect("fd space is not exhausted in tests");
    let dup = fs.dup(fd1).expect("dup");
    // An unrelated symbolic file must survive the demotion.
    fs.register_file_content("/tmp/other", sym_file_bytes(2, "demother"));
    let other = fs.open("/tmp/other".to_string(), FdFlags::ReadOnly).expect("fd space is not exhausted in tests");

    assert!(fs.demote_symbolic_content(fd1), "demotion reported");
    for fd in [fd1, fd2, dup] {
        assert!(fs.fd_content_sym(fd).is_none(), "fd {fd} demoted");
    }
    assert!(
        fs.file_content_for_path("/tmp/d").is_none(),
        "registry entry gone"
    );
    // A fresh open no longer attaches — Python owns the file from here.
    let fd3 = fs.open("/tmp/d".to_string(), FdFlags::ReadOnly).expect("fd space is not exhausted in tests");
    assert!(fs.fd_content_sym(fd3).is_none());
    // Second demotion is a no-op (nothing left to demote).
    assert!(!fs.demote_symbolic_content(fd1));
    // Unrelated file untouched.
    assert!(fs.fd_content_sym(other).is_some());
    assert!(fs.file_content_for_path("/tmp/other").is_some());
    // Plain fds report false (the cheap common case).
    let plain = fs.open("plain.txt".to_string(), FdFlags::WriteOnly).expect("fd space is not exhausted in tests");
    assert!(!fs.demote_symbolic_content(plain));
}

/// angr-qluof: a native write-demotion records the cwd-normalized path in
/// the persistent `demoted_paths` set, and `demote_path` re-applies the
/// demotion on a fresh re-registration (the Python re-add correction).
#[test]
fn test_demoted_paths_tracking_and_re_demote() {
    let mut fs = FileSystem::default();
    fs.register_file_content("/tmp/d", sym_file_bytes(4, "qd"));
    let fd = fs.open("/tmp/d".to_string(), FdFlags::ReadWrite).expect("fd space is not exhausted in tests");
    assert!(fs.demoted_paths().is_empty(), "nothing demoted yet");
    assert!(fs.demote_symbolic_content(fd));
    assert_eq!(
        fs.demoted_paths(),
        vec!["/tmp/d".to_string()],
        "demoted path recorded (normalized)"
    );
    // A Python re-add re-registers the content (re-arming the demotion)...
    fs.register_file_content("/tmp/d", sym_file_bytes(4, "qd2"));
    assert!(
        fs.file_content_for_path("/tmp/d").is_some(),
        "re-registered"
    );
    // ...and the re-add correction (relative spelling) drops it again.
    assert!(
        fs.demote_path("tmp/d"),
        "re-demote cleared re-armed content"
    );
    assert!(
        fs.file_content_for_path("/tmp/d").is_none(),
        "demoted again"
    );
    // Idempotent: re-demoting an already-clean path reports no change but
    // keeps the path in the set.
    assert!(!fs.demote_path("/tmp/d"));
    assert_eq!(fs.demoted_paths(), vec!["/tmp/d".to_string()]);
    // The set survives a snapshot round-trip (worker migration).
    let data = FileSystemData::from(fs);
    let fs2: FileSystem = data.into();
    assert_eq!(fs2.demoted_paths(), vec!["/tmp/d".to_string()]);
}

/// Registration AFTER open leaves the fd without `content_sym` but stamps
/// its `registry_key`, so the registry stays populated yet a write through
/// that fd still drops the entry (or a later open would serve stale
/// content natively).
#[test]
fn test_demote_registry_only_entry() {
    let mut fs = FileSystem::default();
    let fd = fs.open("/tmp/late".to_string(), FdFlags::ReadWrite).expect("fd space is not exhausted in tests");
    fs.register_file_content("/tmp/late", sym_file_bytes(2, "late"));
    assert!(fs.fd_content_sym(fd).is_none());
    assert!(fs.demote_symbolic_content(fd));
    assert!(fs.file_content_for_path("/tmp/late").is_none());
}

/// cwd-drift (angr-0xyq2 A1): the registry key is frozen on the descriptor
/// at attach time, so a guest `chdir` between open and write cannot decouple
/// the registry entry or differently-spelled sibling fds from the demotion.
#[test]
fn test_demote_survives_cwd_drift() {
    let mut fs = FileSystem::default();
    fs.set_cwd(b"/a".to_vec());
    // Relative registration + opens key under cwd-at-the-time: "/a/f".
    fs.register_file_content("f", sym_file_bytes(3, "drift"));
    let fd_rel = fs.open("f".to_string(), FdFlags::ReadWrite).expect("fd space is not exhausted in tests");
    let fd_abs = fs.open("/a/f".to_string(), FdFlags::ReadOnly).expect("fd space is not exhausted in tests");
    assert!(fs.fd_content_sym(fd_rel).is_some());
    assert!(fs.fd_content_sym(fd_abs).is_some());

    // Guest chdir: re-normalizing "f" now would yield "/b/f" and miss both
    // the registry entry and the sibling — registry_key must not care.
    fs.set_cwd(b"/b".to_vec());
    assert!(fs.demote_symbolic_content(fd_rel), "demotion reported");
    assert!(fs.fd_content_sym(fd_rel).is_none(), "written fd demoted");
    assert!(fs.fd_content_sym(fd_abs).is_none(), "sibling fd demoted");
    assert!(
        fs.file_content_for_path("/a/f").is_none(),
        "registry entry gone despite the cwd drift"
    );
    // The choke point agrees: a concrete write on the demoted fd proceeds.
    assert!(fs.write(fd_rel, b"ok"));
}

/// The write choke point itself demotes when a caller skips its per-site
/// gate: nothing is written, the registry and all siblings are dropped.
#[test]
fn test_write_choke_point_refuses_and_demotes() {
    let mut fs = FileSystem::default();
    fs.register_file_content("/tmp/choke", sym_file_bytes(2, "choke"));
    let fd = fs.open("/tmp/choke".to_string(), FdFlags::ReadWrite).expect("fd space is not exhausted in tests");

    // Zero-length writes are a POSIX no-op: accepted, NOT demoted (A3).
    assert!(fs.write(fd, b""));
    assert!(fs.write_at(fd, 5, b""));
    assert!(
        fs.fd_content_sym(fd).is_some(),
        "no-op write must not demote"
    );

    // Non-empty write: refused, demoted, nothing written.
    assert!(!fs.write(fd, b"x"));
    assert!(fs.fd_content_sym(fd).is_none());
    assert!(fs.file_content_for_path("/tmp/choke").is_none());
    assert_eq!(fs.fd_content(fd), b"", "refused write must not mutate");
    // Post-demotion writes are plain concrete writes again.
    assert!(fs.write(fd, b"x"));

    // write_at leg.
    fs.register_file_content("/tmp/choke2", sym_file_bytes(2, "choke2"));
    let fd2 = fs.open("/tmp/choke2".to_string(), FdFlags::ReadWrite).expect("fd space is not exhausted in tests");
    assert!(!fs.write_at(fd2, 1, b"y"));
    assert!(fs.fd_content_sym(fd2).is_none());
    assert_eq!(fs.fd_content(fd2), b"");
}

/// angr-c7xno.67: a guest-controlled write offset past `MAX_FS_FILE_SIZE`
/// must be refused (bounce to Python) instead of driving a multi-exabyte
/// `Vec::resize` — which, under `panic = "abort"`, aborts the whole process.
#[test]
fn test_write_offset_past_size_cap_is_refused() {
    let mut fs = FileSystem::default();
    let fd = fs.open("/tmp/huge".to_string(), FdFlags::ReadWrite).expect("fd space is not exhausted in tests");
    assert!(fs.write(fd, b"seed"));

    // The `lseek(fd, huge, SEEK_SET); write(fd, buf, 1)` chain.
    assert_eq!(fs.seek(fd, i64::MAX, 0), Some(i64::MAX as u64));
    assert!(!fs.write(fd, b"x"), "oversized write must be refused");
    assert_eq!(fs.fd_content(fd), b"seed", "refused write must not mutate");

    // The `pwrite64(fd, buf, 1, huge)` chain.
    assert!(!fs.write_at(fd, u64::MAX, b"x"));
    assert!(!fs.write_at(fd, MAX_FS_FILE_SIZE, b"x"));
    assert_eq!(fs.fd_content(fd), b"seed");

    // Exactly at the cap is still allowed; one byte past is not. Checked on
    // a fresh fd so the multi-MiB buffer is dropped with it.
    let fd2 = fs.open("/tmp/atcap".to_string(), FdFlags::ReadWrite).expect("fd space is not exhausted in tests");
    assert!(fs.write_at(fd2, MAX_FS_FILE_SIZE - 1, b"x"));
    assert_eq!(fs.fd_content(fd2).len(), MAX_FS_FILE_SIZE as usize);
    assert!(!fs.write_at(fd2, MAX_FS_FILE_SIZE - 1, b"xy"));
    assert_eq!(fs.fd_content(fd2).len(), MAX_FS_FILE_SIZE as usize);
}

/// angr-c7xno.67 companion: an oversized write on a bounded-symbolic-content
/// fd must still DEMOTE before refusing. Refusing first would leave Rust
/// serving the registered symbolic content while the Python fallback performs
/// the write — the exact divergence the angr-0xyq2 Phase 2 choke point exists
/// to prevent.
#[test]
fn test_write_offset_cap_still_demotes_symbolic_content() {
    let mut fs = FileSystem::default();
    fs.register_file_content("/tmp/capsym", sym_file_bytes(2, "capsym"));
    let fd = fs.open("/tmp/capsym".to_string(), FdFlags::ReadWrite).expect("fd space is not exhausted in tests");
    assert!(fs.has_content_sym(fd));

    assert!(!fs.write_at(fd, MAX_FS_FILE_SIZE, b"x"));
    assert!(
        fs.fd_content_sym(fd).is_none(),
        "oversized write must demote before refusing"
    );
    assert!(fs.file_content_for_path("/tmp/capsym").is_none());
    assert_eq!(fs.fd_content(fd), b"", "refused write must not mutate");
}

/// B2 (angr-0xyq2): `close` only flips the flag — `content_sym` stays
/// attached — but a closed fd must not serve.
#[test]
fn test_read_sym_closed_fd_returns_none() {
    let mut fs = FileSystem::default();
    fs.register_file_content("/tmp/closed", sym_file_bytes(3, "closed"));
    let fd = fs.open("/tmp/closed".to_string(), FdFlags::ReadOnly).expect("fd space is not exhausted in tests");
    assert!(fs.has_content_sym(fd));
    assert!(fs.close(fd));
    // The content is still attached (fd_content_sym is flag-blind) ...
    assert!(fs.fd_content_sym(fd).is_some());
    // ... but neither primitive serves it, and the predicate agrees.
    assert!(fs.read_sym(fd, 1).is_none());
    assert!(fs.read_sym_at(fd, 0, 1).is_none());
    assert!(!fs.has_content_sym(fd));
}

/// angr-9ke6b.118: `read`/`read_at` guard on `is_open`, so `write`/`write_at`
/// must too — otherwise a closed-but-tracked fd silently accumulates bytes
/// that can never be read back. A closed fd is refused WITHOUT demoting the
/// file's symbolic content (the write is an `EBADF` that never reaches the
/// file, so sibling fds keep serving natively).
#[test]
fn test_write_to_closed_fd_is_refused() {
    let mut fs = FileSystem::default();
    let fd = fs.open("/tmp/wclosed".to_string(), FdFlags::ReadWrite).expect("fd space is not exhausted in tests");
    assert!(fs.write(fd, b"open"));
    assert!(fs.close(fd));

    // Both write primitives refuse, and neither mutates the surviving buffer.
    assert!(!fs.write(fd, b"after-close"));
    assert!(!fs.write_at(fd, 0, b"AFTER"));
    assert_eq!(
        fs.fd_content(fd),
        b"open",
        "a closed fd must not accumulate writes"
    );
    // Zero-length writes stay a POSIX no-op even on a closed fd.
    assert!(fs.write(fd, b""));
    assert!(fs.write_at(fd, 0, b""));

    // A *missing* fd is still auto-vivified (the `write_fd(2, ..)` path).
    assert!(fs.write(4242, b"vivified"));
    assert_eq!(fs.fd_content(4242), b"vivified");

    // Refusing a closed fd must not demote the file's symbolic content:
    // a sibling fd on the same file keeps serving natively.
    fs.register_file_content("/tmp/wclosed_sym", sym_file_bytes(3, "wclosed"));
    let a = fs.open("/tmp/wclosed_sym".to_string(), FdFlags::ReadWrite).expect("fd space is not exhausted in tests");
    let b = fs.open("/tmp/wclosed_sym".to_string(), FdFlags::ReadOnly).expect("fd space is not exhausted in tests");
    assert!(fs.close(a));
    assert!(!fs.write(a, b"x"));
    assert!(
        fs.has_content_sym(b),
        "closed-fd refusal must not demote the file"
    );
}

/// A4 insurance (angr-0xyq2): a symbolic/unresolvable write fd could alias
/// any registered file — `demote_all_symbolic_content` clears every
/// registry entry and every fd's attachment in one shot.
#[test]
fn test_demote_all_symbolic_content_clears_everything() {
    let mut fs = FileSystem::default();
    // Nothing attached: O(1) no-op.
    assert!(!fs.demote_all_symbolic_content());

    fs.register_file_content("/tmp/a", sym_file_bytes(2, "alla"));
    fs.register_file_content("/tmp/b", sym_file_bytes(3, "allb"));
    let fd_a = fs.open("/tmp/a".to_string(), FdFlags::ReadWrite).expect("fd space is not exhausted in tests");
    let fd_b = fs.open("/tmp/b".to_string(), FdFlags::ReadOnly).expect("fd space is not exhausted in tests");
    let dup_b = fs.dup(fd_b).expect("dup");

    assert!(fs.demote_all_symbolic_content());
    for fd in [fd_a, fd_b, dup_b] {
        assert!(fs.fd_content_sym(fd).is_none(), "fd {fd} demoted");
    }
    assert!(fs.file_content_for_path("/tmp/a").is_none());
    assert!(fs.file_content_for_path("/tmp/b").is_none());
    // Fresh opens no longer attach; a second sweep is a no-op.
    let fd_c = fs.open("/tmp/a".to_string(), FdFlags::ReadOnly).expect("fd space is not exhausted in tests");
    assert!(fs.fd_content_sym(fd_c).is_none());
    assert!(!fs.demote_all_symbolic_content());
}

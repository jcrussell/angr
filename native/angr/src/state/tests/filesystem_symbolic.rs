//! Symbolic file content: `register_file_content` attach paths (absolute and
//! cwd-relative), the `content_sym` Arc sharing/serde/snapshot contracts,
//! effective-length accounting when a concrete write runs past the symbolic
//! end, and the `read_sym*` serving/clamping/position rules.
//!
//! Carved out of the former monolithic `state_tests.rs` (angr-c7xno.69).

use super::super::*;
use super::helpers::sym_file_bytes;

// angr-0xyq2 Phase 1: bounded symbolic file content (`content_sym` on the
// fd + path-keyed `file_contents` registry). Data model only — read/fread
// serve paths still consume the concrete buffer; these tests cover the
// registry→open attach, cwd normalization, Arc sharing on fork, the
// length consumers (SEEK_END / stat sizes), and the snapshot wire format
// (including pre-angr-0xyq2 backward compat).

#[test]
fn test_register_file_content_open_attaches() {
    let mut fs = FileSystem::default();
    fs.register_file_content("/tmp/flag", sym_file_bytes(4, "attach"));
    // Registration alone makes the path visible to access/stat.
    assert!(fs.is_path_known("/tmp/flag"));

    let fd = fs.open("/tmp/flag".to_string(), FdFlags::ReadOnly);
    let attached = fs
        .fd_content_sym(fd)
        .expect("open attaches registry content");
    let registered = fs.file_content_for_path("/tmp/flag").expect("registered");
    // Shared Arc — refcount bump, not a deep BV-vector clone.
    assert!(Arc::ptr_eq(&attached, &registered));

    // Length consumers see the symbolic byte count (concrete buffer is empty).
    assert_eq!(fs.effective_size(fd), Some(4));
    assert_eq!(fs.content_size_for_path("/tmp/flag"), Some(4));
    // Read-serve state is untouched: the concrete buffer stays empty.
    assert_eq!(fs.fd_content(fd), b"");

    // open_symbolic does NOT attach: the stream model (mint-forever) and
    // the bounded-file model (EOF at size) have conflicting EOF semantics,
    // and the stream model wins for open_symbolic (angr-0xyq2 A5).
    let fd_sym = fs.open_symbolic("/tmp/flag".to_string(), FdFlags::ReadOnly);
    assert!(fs.is_symbolic(fd_sym));
    assert!(fs.fd_content_sym(fd_sym).is_none());
}

#[test]
fn test_register_file_content_cwd_relative_open() {
    let mut fs = FileSystem::default();
    fs.set_cwd(b"/home/user".to_vec());
    // Relative registration keys under the cwd-normalized absolute path.
    fs.register_file_content("flag.txt", sym_file_bytes(3, "rel"));
    assert!(fs.is_path_known("/home/user/flag.txt"));

    // Both the relative and the absolute spelling of the same file attach.
    let fd_rel = fs.open("flag.txt".to_string(), FdFlags::ReadOnly);
    let fd_abs = fs.open("/home/user/flag.txt".to_string(), FdFlags::ReadOnly);
    let rel = fs.fd_content_sym(fd_rel).expect("relative open attaches");
    let abs = fs.fd_content_sym(fd_abs).expect("absolute open attaches");
    assert!(Arc::ptr_eq(&rel, &abs));
}

#[test]
fn test_content_sym_fork_shares_arc() {
    let mut fs = FileSystem::default();
    fs.register_file_content("/tmp/flag", sym_file_bytes(4, "fork"));
    let fd = fs.open("/tmp/flag".to_string(), FdFlags::ReadOnly);

    let handle = fs.fd_content_sym(fd).expect("attached");
    let before = Arc::strong_count(&handle);
    let mut forked = fs.clone();
    // The clone itself is O(1) (shared fd-map Arc): no new content refs yet.
    assert_eq!(Arc::strong_count(&handle), before);
    // A position advance forces CoW on the fork's fd map — the descriptor
    // clone must bump the content Arc refcount, not deep-clone the BVs.
    forked.seek(fd, 1, 0);
    assert_eq!(Arc::strong_count(&handle), before + 1);
    assert!(Arc::ptr_eq(&handle, &forked.fd_content_sym(fd).unwrap()));
}

#[test]
fn test_seek_end_uses_effective_len() {
    let mut fs = FileSystem::default();
    fs.register_file_content("/tmp/flag", sym_file_bytes(10, "seek"));
    let fd = fs.open("/tmp/flag".to_string(), FdFlags::ReadOnly);

    // Concrete buffer is empty; SEEK_END must key off the symbolic length.
    assert_eq!(fs.seek(fd, 0, 2), Some(10));
    assert_eq!(fs.seek(fd, -3, 2), Some(7));
    // SEEK_SET / SEEK_CUR are unaffected.
    assert_eq!(fs.seek(fd, 2, 0), Some(2));
    assert_eq!(fs.seek(fd, 1, 1), Some(3));
}

/// Fix 3 (angr-0xyq2 review): a concrete buffer grown past the symbolic end
/// must not be masked by `content_sym` — `effective_len` is the max of both
/// legs, mirroring Python `SimFile.write` size semantics. The Phase 2 write
/// choke point refuses (and demotes) concrete writes on attached fds, so
/// the mixed state can no longer arise via `FileSystem::write` — build it
/// through the serde shadow (`FileSystemData`) to keep pinning the
/// defense-in-depth `max()`.
#[test]
fn test_effective_len_concrete_write_past_symbolic_end() {
    let mut fs = FileSystem::default();
    fs.register_file_content("/tmp/grow", sym_file_bytes(4, "grow"));
    let fd = fs.open("/tmp/grow".to_string(), FdFlags::ReadOnly);
    assert_eq!(fs.effective_size(fd), Some(4));

    // Concrete content of 10 bytes: its end (10) exceeds the symbolic byte
    // count (4) — the larger concrete length must win.
    let mut data = FileSystemData::from(fs);
    data.fds.get_mut(&fd).expect("fd on the wire").content = b"0123456789".to_vec();
    let mut fs = FileSystem::from(data);
    assert_eq!(fs.effective_size(fd), Some(10));
    assert_eq!(fs.content_size_for_path("/tmp/grow"), Some(10));
    assert_eq!(fs.seek(fd, 0, 2), Some(10));

    // The symbolic content itself is untouched (Phase 2 write-demotion is
    // the primary defense; this max is defense-in-depth).
    assert_eq!(fs.fd_content_sym(fd).unwrap().len(), 4);
}

/// Fix 1 (angr-0xyq2 review): `content_size_for_path` must consult the
/// registry, so a registered-but-never-opened path reports its content
/// length (unit leg; the stat syscall leg lives in file_path_tests.rs).
#[test]
fn test_content_size_for_path_registry_without_fd() {
    let mut fs = FileSystem::default();
    fs.register_file_content("/tmp/noopen", sym_file_bytes(6, "noopen"));
    assert_eq!(fs.content_size_for_path("/tmp/noopen"), Some(6));
    // Relative spelling of the same path agrees (both legs normalize).
    assert_eq!(fs.content_size_for_path("tmp/noopen"), Some(6));
    // Content-less register_known_path still reports None (stat's zero-size
    // default path — pinned by stat_registered_path_without_fd_uses_zero_size).
    fs.register_known_path("/etc/registered-only".to_string());
    assert_eq!(fs.content_size_for_path("/etc/registered-only"), None);
}

#[test]
fn test_open_without_registry_entry_unchanged() {
    let mut fs = FileSystem::default();
    let fd = fs.open("plain.txt".to_string(), FdFlags::ReadOnly);
    assert!(fs.fd_content_sym(fd).is_none());
    assert_eq!(fs.effective_size(fd), Some(0));
    assert_eq!(fs.seek(fd, 0, 2), Some(0));

    let fd2 = fs.open_with_content("c.txt".to_string(), FdFlags::ReadOnly, b"abc".to_vec());
    assert!(fs.fd_content_sym(fd2).is_none());
    // effective_len falls back to the concrete buffer length.
    assert_eq!(fs.effective_size(fd2), Some(3));
    assert_eq!(fs.content_size_for_path("c.txt"), Some(3));
}

#[test]
fn test_content_sym_serde_roundtrip() {
    let mut fs = FileSystem::default();
    let mut bytes = sym_file_bytes(2, "wire");
    bytes.push(RustBV::concrete(0x41, 8)); // mixed file: concrete tail byte
    fs.register_file_content("/tmp/flag", bytes);
    let fd = fs.open("/tmp/flag".to_string(), FdFlags::ReadOnly);

    let json = serde_json::to_string(&fs).expect("serialize");
    let restored: FileSystem = serde_json::from_str(&json).expect("deserialize");

    // Symbolic leaves rebuild by (name, width) via the RustBVData shadow.
    let content = restored.fd_content_sym(fd).expect("content_sym survives");
    assert_eq!(content.len(), 3);
    match &content[0] {
        RustBV::Symbolic { name, width, .. } => {
            assert_eq!(&**name, "wire_0");
            assert_eq!(*width, 8);
        }
        other => panic!("expected Symbolic, got {other:?}"),
    }
    assert!(matches!(
        &content[2],
        RustBV::Concrete {
            value: 0x41,
            width: 8
        }
    ));
    // The path-keyed registry survives independently of the fd.
    assert_eq!(
        restored
            .file_content_for_path("/tmp/flag")
            .expect("registry survives")
            .len(),
        3
    );
    assert_eq!(restored.effective_size(fd), Some(3));
}

#[test]
fn test_content_sym_snapshot_backward_compat() {
    let mut fs = FileSystem::default();
    fs.register_file_content("/tmp/flag", sym_file_bytes(2, "compat"));
    let fd = fs.open("/tmp/flag".to_string(), FdFlags::ReadOnly);

    // Simulate a pre-angr-0xyq2 snapshot: strip the new fields entirely.
    let mut v = serde_json::to_value(&fs).expect("to_value");
    v.as_object_mut().unwrap().remove("file_contents");
    for fd_v in v["fds"].as_object_mut().unwrap().values_mut() {
        fd_v.as_object_mut().unwrap().remove("content_sym");
    }

    let restored: FileSystem = serde_json::from_value(v).expect("old snapshot loads");
    assert!(restored.fd_content_sym(fd).is_none());
    assert!(restored.file_content_for_path("/tmp/flag").is_none());
    // effective_len falls back to the (empty) concrete buffer.
    assert_eq!(restored.effective_size(fd), Some(0));
}

// angr-0xyq2 Phase 2: native serving (`read_sym` / `read_sym_at`) and
// write-demotion (`demote_symbolic_content`). The procedure/syscall wiring
// legs live in the respective *_tests.rs files; these pin the FileSystem
// primitives.

#[test]
fn test_read_sym_serves_clamps_and_advances() {
    let mut fs = FileSystem::default();
    let content = sym_file_bytes(5, "rs");
    fs.register_file_content("/tmp/f", content.clone());
    let fd = fs.open("/tmp/f".to_string(), FdFlags::ReadOnly);

    // First read: the served BVs are the registered entries, in order.
    // (Debug compare: RustBV's PartialEq is concrete-only by design, so
    // symbolic leaves never compare equal structurally.)
    let first = fs.read_sym(fd, 3).expect("content_sym attached");
    assert_eq!(format!("{first:?}"), format!("{:?}", &content[..3]));
    assert_eq!(fs.fd_info(fd).unwrap().1, 3, "position advanced");

    // Over-long read clamps to the remainder (Python max(0, min(...))).
    let rest = fs.read_sym(fd, 10).expect("content_sym attached");
    assert_eq!(format!("{rest:?}"), format!("{:?}", &content[3..]));

    // EOF: Some(empty) — the caller returns 0 — and position stays put.
    assert_eq!(fs.read_sym(fd, 4).expect("still attached"), Vec::new());
    assert_eq!(fs.fd_info(fd).unwrap().1, 5);

    // Fds without content_sym return None (caller keeps concrete logic).
    let plain = fs.open_with_content("p.txt".to_string(), FdFlags::ReadOnly, b"ab".to_vec());
    assert!(fs.read_sym(plain, 1).is_none());
}

/// A position beyond `content_sym.len()` (e.g. an unchecked `SEEK_SET`)
/// must serve 0 bytes (slice within symbolic bounds only), not panic.
#[test]
fn test_read_sym_position_past_symbolic_end_serves_empty() {
    let mut fs = FileSystem::default();
    fs.register_file_content("/tmp/mixed", sym_file_bytes(4, "mixededge"));
    let fd = fs.open("/tmp/mixed".to_string(), FdFlags::ReadOnly);
    assert_eq!(fs.seek(fd, 6, 0), Some(6)); // SEEK_SET does not clamp to len
    assert_eq!(
        fs.read_sym(fd, 3).expect("content_sym attached"),
        Vec::new()
    );
    assert_eq!(
        fs.fd_info(fd).unwrap().1,
        6,
        "position untouched at sym-EOF"
    );
}

#[test]
fn test_read_sym_at_does_not_move_position() {
    let mut fs = FileSystem::default();
    let content = sym_file_bytes(4, "rsat");
    fs.register_file_content("/tmp/pread", content.clone());
    let fd = fs.open("/tmp/pread".to_string(), FdFlags::ReadOnly);

    // Debug compare — see test_read_sym_serves_clamps_and_advances.
    let got = fs.read_sym_at(fd, 1, 2).expect("content_sym attached");
    assert_eq!(format!("{got:?}"), format!("{:?}", &content[1..3]));
    assert_eq!(
        fs.fd_info(fd).unwrap().1,
        0,
        "pread semantics: position unmoved"
    );
    // Past-EOF offset serves empty, no panic.
    assert_eq!(fs.read_sym_at(fd, 9, 2).expect("attached"), Vec::new());
    // No content_sym → None.
    let plain = fs.open_with_content("q.txt".to_string(), FdFlags::ReadOnly, b"ab".to_vec());
    assert!(fs.read_sym_at(plain, 0, 1).is_none());
}

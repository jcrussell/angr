// angr-c2cv: RustSimState unit tests, extracted out of the former in-file
// `mod tests` (~590 lines) into a sibling file to shrink state.rs below the
// god-object threshold. Declared as a direct child of the crate-root state
// module so `use super::*` reaches the module's private items.

use super::*;

#[test]
fn test_state_creation() {
    let state = RustSimState::new("amd64").unwrap();
    assert_eq!(state.vex_arch(), VexArch::AMD64);
    assert_eq!(state.pc(), 0);
}

#[test]
fn test_state_fork() {
    let mut state1 = RustSimState::new("amd64").unwrap();
    state1.set_pc(0x1000);
    state1.set_register("rax", RustBV::concrete(42, 64));

    let state2 = state1.fork();

    // Both should have same values
    assert_eq!(state2.pc(), 0x1000);
    assert_eq!(state2.get_register("rax").unwrap().as_u64(), Some(42));

    // Different state IDs
    assert_ne!(state1.state_id(), state2.state_id());

    // state2's parent should be state1
    assert_eq!(state2.parent_id(), Some(state1.state_id()));
}

#[test]
fn test_getopt_cursor_default_and_fork_isolation() {
    // Foundation slice for native getopt parity (bead angr-bhk0a): the
    // per-state getopt cursor must default to (optind=1, optchar=0) — the
    // glibc/Python `state.libc.getopt_optind`/`getopt_optchar` defaults — and
    // be copied (not shared) across fork so each path scans argv independently.
    let mut parent = RustSimState::new("amd64").unwrap();
    assert_eq!(parent.getopt_cursor(), (1, 0));

    parent.set_getopt_cursor(4, 2);
    let mut child = parent.fork();
    assert_eq!(child.getopt_cursor(), (4, 2));

    // Mutating the child must not disturb the parent (CoW isolation).
    child.set_getopt_cursor(9, 0);
    assert_eq!(child.getopt_cursor(), (9, 0));
    assert_eq!(parent.getopt_cursor(), (4, 2));
}

#[test]
fn test_getopt_extern_addrs_default_and_fork_isolation() {
    // bhk0a.1: the loader-resolved getopt(3) extern-global addresses
    // (optind/optarg/optopt) default to None (no init-push yet -> native proc
    // defers to Python) and are copied (not shared) across fork.
    let mut parent = RustSimState::new("amd64").unwrap();
    let d = parent.getopt_extern();
    assert_eq!((d.optind, d.optarg, d.optopt), (None, None, None));

    parent.set_getopt_extern(GetoptExternAddrs {
        optind: Some(0x601000),
        optarg: Some(0x601008),
        optopt: Some(0x601010),
    });
    let mut child = parent.fork();
    let c = child.getopt_extern();
    assert_eq!(
        (c.optind, c.optarg, c.optopt),
        (Some(0x601000), Some(0x601008), Some(0x601010))
    );

    // CoW isolation: mutating the child must not disturb the parent.
    child.set_getopt_extern(GetoptExternAddrs::default());
    assert_eq!(child.getopt_extern().optind, None);
    assert_eq!(parent.getopt_extern().optind, Some(0x601000));
}

#[test]
fn test_native_resume_stack_default_and_fork_isolation() {
    // angr-pn3w8 (S1): the native sub-call resume stack defaults to empty and
    // is copied (not shared) across fork so each path resumes its own pending
    // sub-calls. No dispatcher yet — this only exercises the per-state plumbing.
    let mut parent = RustSimState::new("amd64").unwrap();
    assert!(parent.native_resume_stack().is_empty());

    parent.push_native_resume_frame(NativeResumeFrame {
        proc_name: "pthread_once".to_string(),
        resume_tag: 0,
        saved_args: vec![
            RustBV::concrete(0x601000, 64),
            RustBV::concrete(0x400500, 64),
        ],
        caller_return_addr: 0x400600,
    });
    assert_eq!(parent.native_resume_stack().len(), 1);

    let mut child = parent.fork();
    assert_eq!(child.native_resume_stack().len(), 1);
    let frame = &child.native_resume_stack()[0];
    assert_eq!(frame.proc_name, "pthread_once");
    assert_eq!(frame.resume_tag, 0);
    assert_eq!(frame.saved_args.len(), 2);
    assert_eq!(frame.saved_args[0].as_u64(), Some(0x601000));

    // CoW isolation: popping in the child must not disturb the parent's stack.
    let popped = child.pop_native_resume_frame();
    assert!(popped.is_some());
    assert!(child.native_resume_stack().is_empty());
    assert_eq!(parent.native_resume_stack().len(), 1);
}

#[test]
fn test_set_detailed_history_honors_cap() {
    // set_detailed_history (called once per step from interpreter results)
    // must drain the oldest entries when the incoming buffer exceeds the
    // configured cap. Otherwise long blocks bypass max_history.
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_max_history(3);
    let entries: Vec<HistoryEntry> = (0..10)
        .map(|i| HistoryEntry {
            addr: 0x1000 + i,
            jumpkind: 0,
            jump_target: 0,
        })
        .collect();
    state.set_detailed_history(entries);
    let kept = state.detailed_history();
    assert_eq!(kept.len(), 3);
    // FIFO eviction: should retain the most-recent 3 entries.
    assert_eq!(kept[0].addr, 0x1007);
    assert_eq!(kept[2].addr, 0x1009);
}

#[test]
fn test_set_max_history_trims_retroactively() {
    // Lowering max_history on a state that already exceeds the new cap
    // must FIFO-evict the oldest entries down to the cap immediately.
    // Without this, add_to_history (which only removes one entry per
    // push when over cap) never converges and the buffer stays bloated.
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_max_history(0); // unlimited
    for i in 0..20u64 {
        state.add_to_history(0x3000 + i);
        state.add_history_entry(0x3000 + i, 0, 0);
    }
    assert_eq!(state.history().len(), 20);
    assert_eq!(state.detailed_history().len(), 20);

    // Retroactively cap to 4 — both buffers shrink to the most-recent 4.
    state.set_max_history(4);
    let kept = state.history();
    assert_eq!(kept.len(), 4);
    assert_eq!(
        kept.iter().copied().collect::<Vec<_>>(),
        vec![0x3010, 0x3011, 0x3012, 0x3013]
    );
    let kept_detailed = state.detailed_history();
    assert_eq!(kept_detailed.len(), 4);
    assert_eq!(kept_detailed[0].addr, 0x3010);
    assert_eq!(kept_detailed[3].addr, 0x3013);
}

#[test]
fn test_set_max_history_zero_no_trim() {
    // Switching to max=0 (unlimited) must NOT trim — existing entries stay.
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_max_history(5);
    for i in 0..5u64 {
        state.add_to_history(0x4000 + i);
    }
    state.set_max_history(0);
    assert_eq!(state.history().len(), 5);
}

#[test]
fn test_set_detailed_history_unlimited() {
    // max_history = 0 means no cap (legacy behavior).
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_max_history(0);
    let entries: Vec<HistoryEntry> = (0..50)
        .map(|i| HistoryEntry {
            addr: 0x2000 + i,
            jumpkind: 0,
            jump_target: 0,
        })
        .collect();
    state.set_detailed_history(entries);
    assert_eq!(state.detailed_history().len(), 50);
}

#[test]
fn test_state_memory() {
    let mut state = RustSimState::new("amd64").unwrap();

    // Map and write
    state.map_memory(0x1000, 0x1000, Permission::RWX);
    state
        .memory_store(0x1000, RustBV::concrete(0xDEADBEEF, 32))
        .unwrap();

    // Read back
    let val = state.memory_load(0x1000, 4).unwrap();
    assert_eq!(val.as_u64(), Some(0xDEADBEEF));
}

#[test]
fn test_state_fork_memory_cow() {
    let mut state1 = RustSimState::new("amd64").unwrap();
    state1.map_memory(0x1000, 0x1000, Permission::RWX);
    state1
        .memory_store(0x1000, RustBV::concrete(0xAAAA, 16))
        .unwrap();

    let mut state2 = state1.fork();

    // Modify state2
    state2
        .memory_store(0x1000, RustBV::concrete(0xBBBB, 16))
        .unwrap();

    // state1 should still have original value
    let val1 = state1.memory_load(0x1000, 2).unwrap();
    assert_eq!(val1.as_u64(), Some(0xAAAA));

    // state2 should have new value
    let val2 = state2.memory_load(0x1000, 2).unwrap();
    assert_eq!(val2.as_u64(), Some(0xBBBB));
}

#[test]
fn test_filesystem_default() {
    let fs = FileSystem::default();
    assert!(fs.is_open(0)); // stdin
    assert!(fs.is_open(1)); // stdout
    assert!(fs.is_open(2)); // stderr
    assert!(!fs.is_open(3));
    assert_eq!(fs.next_fd(), 3);
}

#[test]
fn test_filesystem_open_close() {
    let mut fs = FileSystem::default();
    let fd = fs.open("test.txt".to_string(), FdFlags::ReadOnly);
    assert_eq!(fd, 3);
    assert!(fs.is_open(3));

    let closed = fs.close(3);
    assert!(closed);
    assert!(!fs.is_open(3));

    // Double close returns false
    assert!(!fs.close(3));
}

#[test]
fn test_filesystem_write_read() {
    let mut fs = FileSystem::default();
    let fd = fs.open_with_content(
        "data.bin".to_string(),
        FdFlags::ReadOnly,
        b"hello world".to_vec(),
    );

    let data = fs.read(fd, 5);
    assert_eq!(data, b"hello");

    let data2 = fs.read(fd, 6);
    assert_eq!(data2, b" world");

    // Read past end
    let data3 = fs.read(fd, 10);
    assert!(data3.is_empty());
}

#[test]
fn test_filesystem_read_closed_fd_serves_nothing() {
    // Parity with read_sym/read_sym_at: a closed fd serves no bytes even
    // though its content buffer survives (angr-myzjx.23).
    let mut fs = FileSystem::default();
    let fd = fs.open_with_content(
        "data.bin".to_string(),
        FdFlags::ReadOnly,
        b"hello world".to_vec(),
    );
    assert_eq!(fs.read_at(fd, 0, 5), b"hello");
    assert!(fs.close(fd));

    // Both the position-advancing and positioned reads must refuse a closed fd.
    assert!(fs.read(fd, 5).is_empty());
    assert!(fs.read_at(fd, 0, 5).is_empty());
}

#[test]
fn test_filesystem_seek() {
    let mut fs = FileSystem::default();
    let fd = fs.open_with_content("data.bin".to_string(), FdFlags::ReadOnly, vec![0u8; 100]);

    // SEEK_SET
    assert_eq!(fs.seek(fd, 50, 0), Some(50));
    // SEEK_CUR
    assert_eq!(fs.seek(fd, 10, 1), Some(60));
    // SEEK_END
    assert_eq!(fs.seek(fd, -5, 2), Some(95));
    // Invalid whence
    assert_eq!(fs.seek(fd, 0, 99), None);
}

#[test]
fn test_filesystem_write_position_aware() {
    let mut fs = FileSystem::default();
    let fd = fs.open("out.bin".to_string(), FdFlags::WriteOnly);

    // Sequential writes (no seek): position-aware path is byte-identical to a
    // plain append, and advances the position to EOF each time. `true` =
    // accepted (no bounded symbolic content on the fd).
    assert!(fs.write(fd, b"abc"));
    assert!(fs.write(fd, b"def"));
    assert_eq!(fs.fd_content(fd), b"abcdef");

    // Seek back and overwrite in place (the case append-only would corrupt).
    assert_eq!(fs.seek(fd, 0, 0), Some(0));
    assert!(fs.write(fd, b"XY"));
    assert_eq!(fs.fd_content(fd), b"XYcdef");

    // A subsequent write continues from the advanced position (after "XY").
    assert!(fs.write(fd, b"Z"));
    assert_eq!(fs.fd_content(fd), b"XYZdef");

    // Sparse seek past EOF zero-fills the gap, like write_at/pwrite.
    assert_eq!(fs.seek(fd, 8, 0), Some(8));
    assert!(fs.write(fd, b"!"));
    assert_eq!(fs.fd_content(fd), b"XYZdef\x00\x00!");
}

#[test]
fn test_filesystem_fork_isolation() {
    let mut state = RustSimState::new("amd64").unwrap();
    state
        .file_system()
        .open("test.txt".to_string(), FdFlags::ReadOnly);
    assert!(state.file_system_ref().is_open(3));

    let mut forked = state.fork();
    // Close in forked state
    forked.file_system().close(3);
    assert!(!forked.file_system_ref().is_open(3));
    // Original should still be open
    assert!(state.file_system_ref().is_open(3));
}

// angr-0xyq2 Phase 1: bounded symbolic file content (`content_sym` on the
// fd + path-keyed `file_contents` registry). Data model only — read/fread
// serve paths still consume the concrete buffer; these tests cover the
// registry→open attach, cwd normalization, Arc sharing on fork, the
// length consumers (SEEK_END / stat sizes), and the snapshot wire format
// (including pre-angr-0xyq2 backward compat).

/// `n` fresh 8-bit symbolic bytes named `{prefix}_{i}` (ids sit in
/// 0x5f000+ to stay clear of other symbolic ids in these tests; the
/// prefixes distinguish tests from each other).
fn sym_file_bytes(n: usize, prefix: &str) -> Vec<RustBV> {
    (0..n)
        .map(|i| RustBV::symbolic_with_id(0x5f000 + i as u64, format!("{prefix}_{i}"), 8))
        .collect()
}

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
fn test_filesystem_normalize_path() {
    let mut fs = FileSystem::default();
    assert_eq!(fs.normalize_path("/a/b/../c/./d"), "/a/c/d");
    assert_eq!(fs.normalize_path("/../.."), "/");
    // Truncates at the first NUL, then resolves relative to cwd (/).
    assert_eq!(fs.normalize_path("x\0junk"), "/x");

    fs.set_cwd(b"/home/user".to_vec());
    assert_eq!(fs.normalize_path("f.txt"), "/home/user/f.txt");
    assert_eq!(fs.normalize_path("../f"), "/home/f");
    assert_eq!(fs.normalize_path(""), "/home/user");
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

#[test]
fn test_demote_symbolic_content_clears_all_fds_and_registry() {
    let mut fs = FileSystem::default();
    fs.register_file_content("/tmp/d", sym_file_bytes(4, "dem"));
    // Three handles on the same file: absolute, relative spelling, dup.
    let fd1 = fs.open("/tmp/d".to_string(), FdFlags::ReadWrite);
    let fd2 = fs.open("tmp/d".to_string(), FdFlags::ReadOnly);
    let dup = fs.dup(fd1).expect("dup");
    // An unrelated symbolic file must survive the demotion.
    fs.register_file_content("/tmp/other", sym_file_bytes(2, "demother"));
    let other = fs.open("/tmp/other".to_string(), FdFlags::ReadOnly);

    assert!(fs.demote_symbolic_content(fd1), "demotion reported");
    for fd in [fd1, fd2, dup] {
        assert!(fs.fd_content_sym(fd).is_none(), "fd {fd} demoted");
    }
    assert!(
        fs.file_content_for_path("/tmp/d").is_none(),
        "registry entry gone"
    );
    // A fresh open no longer attaches — Python owns the file from here.
    let fd3 = fs.open("/tmp/d".to_string(), FdFlags::ReadOnly);
    assert!(fs.fd_content_sym(fd3).is_none());
    // Second demotion is a no-op (nothing left to demote).
    assert!(!fs.demote_symbolic_content(fd1));
    // Unrelated file untouched.
    assert!(fs.fd_content_sym(other).is_some());
    assert!(fs.file_content_for_path("/tmp/other").is_some());
    // Plain fds report false (the cheap common case).
    let plain = fs.open("plain.txt".to_string(), FdFlags::WriteOnly);
    assert!(!fs.demote_symbolic_content(plain));
}

/// angr-qluof: a native write-demotion records the cwd-normalized path in
/// the persistent `demoted_paths` set, and `demote_path` re-applies the
/// demotion on a fresh re-registration (the Python re-add correction).
#[test]
fn test_demoted_paths_tracking_and_re_demote() {
    let mut fs = FileSystem::default();
    fs.register_file_content("/tmp/d", sym_file_bytes(4, "qd"));
    let fd = fs.open("/tmp/d".to_string(), FdFlags::ReadWrite);
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
    let fd = fs.open("/tmp/late".to_string(), FdFlags::ReadWrite);
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
    let fd_rel = fs.open("f".to_string(), FdFlags::ReadWrite);
    let fd_abs = fs.open("/a/f".to_string(), FdFlags::ReadOnly);
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
    let fd = fs.open("/tmp/choke".to_string(), FdFlags::ReadWrite);

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
    let fd2 = fs.open("/tmp/choke2".to_string(), FdFlags::ReadWrite);
    assert!(!fs.write_at(fd2, 1, b"y"));
    assert!(fs.fd_content_sym(fd2).is_none());
    assert_eq!(fs.fd_content(fd2), b"");
}

/// B2 (angr-0xyq2): `close` only flips the flag — `content_sym` stays
/// attached — but a closed fd must not serve.
#[test]
fn test_read_sym_closed_fd_returns_none() {
    let mut fs = FileSystem::default();
    fs.register_file_content("/tmp/closed", sym_file_bytes(3, "closed"));
    let fd = fs.open("/tmp/closed".to_string(), FdFlags::ReadOnly);
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
    let fd = fs.open("/tmp/wclosed".to_string(), FdFlags::ReadWrite);
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
    let a = fs.open("/tmp/wclosed_sym".to_string(), FdFlags::ReadWrite);
    let b = fs.open("/tmp/wclosed_sym".to_string(), FdFlags::ReadOnly);
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
    let fd_a = fs.open("/tmp/a".to_string(), FdFlags::ReadWrite);
    let fd_b = fs.open("/tmp/b".to_string(), FdFlags::ReadOnly);
    let dup_b = fs.dup(fd_b).expect("dup");

    assert!(fs.demote_all_symbolic_content());
    for fd in [fd_a, fd_b, dup_b] {
        assert!(fs.fd_content_sym(fd).is_none(), "fd {fd} demoted");
    }
    assert!(fs.file_content_for_path("/tmp/a").is_none());
    assert!(fs.file_content_for_path("/tmp/b").is_none());
    // Fresh opens no longer attach; a second sweep is a no-op.
    let fd_c = fs.open("/tmp/a".to_string(), FdFlags::ReadOnly);
    assert!(fs.fd_content_sym(fd_c).is_none());
    assert!(!fs.demote_all_symbolic_content());
}

#[test]
fn test_filesystem_backward_compat() {
    // fd_buffer/write_fd should still work through FileSystem
    let mut state = RustSimState::new("amd64").unwrap();
    assert!(state.write_stdout(b"hello"));
    assert_eq!(state.stdout_buffer(), b"hello");
    assert!(state.has_stdout());

    assert!(state.write_fd(2, b"err"));
    assert_eq!(state.fd_buffer(2), b"err");
}

#[test]
fn test_filesystem_open_fds() {
    let mut fs = FileSystem::default();
    let open = fs.open_fds();
    assert_eq!(open, vec![0, 1, 2]); // stdin, stdout, stderr

    fs.open("a.txt".to_string(), FdFlags::ReadOnly);
    fs.open("b.txt".to_string(), FdFlags::WriteOnly);
    let open = fs.open_fds();
    assert_eq!(open, vec![0, 1, 2, 3, 4]);

    fs.close(3);
    let open = fs.open_fds();
    assert_eq!(open, vec![0, 1, 2, 4]);
}

#[test]
fn test_filesystem_dup_dup2_pipe() {
    let mut fs = FileSystem::default();
    let fd = fs.open("a.txt".to_string(), FdFlags::ReadOnly);
    assert_eq!(fd, 3);

    // dup
    let dup_fd = fs.dup(fd).unwrap();
    assert_eq!(dup_fd, 4);
    assert_eq!(fs.fd_info(dup_fd).unwrap().0, "a.txt");

    // dup of closed fd returns None
    fs.close(fd);
    assert!(fs.dup(fd).is_none());

    // dup2 with fresh state
    let mut fs2 = FileSystem::default();
    let src = fs2.open("src.txt".to_string(), FdFlags::ReadOnly);
    let dst = fs2.dup2(src, 10).unwrap();
    assert_eq!(dst, 10);
    assert_eq!(fs2.fd_info(10).unwrap().0, "src.txt");
    // next_fd advanced past 10
    assert!(fs2.next_fd() > 10);

    // dup2(self, self) returns self when open
    assert_eq!(fs2.dup2(src, src), Some(src));

    // pipe allocates two consecutive fds
    let mut fs3 = FileSystem::default();
    let (r, w) = fs3.pipe();
    assert_eq!((r, w), (3, 4));
    assert!(fs3.is_open(r));
    assert!(fs3.is_open(w));
    assert_eq!(fs3.fd_info(r).unwrap().2, 0); // ReadOnly
    assert_eq!(fs3.fd_info(w).unwrap().2, 1); // WriteOnly
}

#[test]
fn test_inspection_default_inactive() {
    let mgr = InspectionManager::default();
    assert!(!mgr.is_active());
    assert!(!mgr.is_enabled(InspectEvent::MemRead));
}

#[test]
fn test_inspection_enable_disable() {
    let mut mgr = InspectionManager::default();
    mgr.enable(InspectEvent::MemRead);
    assert!(mgr.is_active());
    assert!(mgr.is_enabled(InspectEvent::MemRead));
    assert!(!mgr.is_enabled(InspectEvent::MemWrite));

    mgr.enable_all();
    assert!(mgr.is_enabled(InspectEvent::Fork));
    assert!(mgr.is_enabled(InspectEvent::Exit));

    mgr.disable(InspectEvent::MemRead);
    assert!(!mgr.is_enabled(InspectEvent::MemRead));
    assert!(mgr.is_enabled(InspectEvent::MemWrite));

    mgr.disable_all();
    assert!(!mgr.is_active());
}

#[test]
fn test_inspection_record_events() {
    let mut mgr = InspectionManager::default();
    mgr.enable(InspectEvent::MemWrite);

    mgr.record(InspectEvent::MemWrite, 0x1000, 4, 0x400000);
    mgr.record(InspectEvent::MemWrite, 0x1004, 8, 0x400010);

    assert_eq!(mgr.events().len(), 2);
    assert_eq!(mgr.event_counts()[InspectEvent::MemWrite as usize], 2);

    let e = &mgr.events()[0];
    assert_eq!(e.event, InspectEvent::MemWrite);
    assert_eq!(e.addr, 0x1000);
    assert_eq!(e.size, 4);
    assert_eq!(e.block_addr, 0x400000);
}

#[test]
fn test_inspection_ring_buffer() {
    let mut mgr = InspectionManager::default();
    mgr.set_max_events(3);
    mgr.enable(InspectEvent::MemRead);

    for i in 0..5 {
        mgr.record(InspectEvent::MemRead, i * 0x100, 4, 0);
    }

    // Only last 3 should remain
    assert_eq!(mgr.events().len(), 3);
    assert_eq!(mgr.events()[0].addr, 0x200);
    assert_eq!(mgr.events()[2].addr, 0x400);
    // But total count should be 5
    assert_eq!(mgr.event_counts()[InspectEvent::MemRead as usize], 5);
}

/// `set_max_events` shrinking drops the OLDEST events, keeping the newest
/// `max`, and a subsequent `record` still honours the new cap.
#[test]
fn test_inspection_set_max_events_shrinks_from_the_front() {
    let mut mgr = InspectionManager::default();
    mgr.enable(InspectEvent::MemRead);
    for i in 0..5 {
        mgr.record(InspectEvent::MemRead, i * 0x100, 4, 0);
    }
    assert_eq!(mgr.events().len(), 5);

    mgr.set_max_events(2);
    assert_eq!(mgr.events().len(), 2);
    assert_eq!(mgr.events()[0].addr, 0x300);
    assert_eq!(mgr.events()[1].addr, 0x400);

    mgr.record(InspectEvent::MemRead, 0x500, 4, 0);
    assert_eq!(mgr.events().len(), 2);
    assert_eq!(mgr.events()[0].addr, 0x400);
    assert_eq!(mgr.events()[1].addr, 0x500);
    assert_eq!(mgr.event_counts()[InspectEvent::MemRead as usize], 6);
}

/// `max_events == 0` means "count but retain nothing" — it must not panic
/// and must not leave a stray entry in the buffer.
#[test]
fn test_inspection_max_events_zero_retains_nothing() {
    let mut mgr = InspectionManager::default();
    mgr.enable(InspectEvent::MemWrite);
    mgr.set_max_events(0);

    mgr.record(InspectEvent::MemWrite, 0x1000, 4, 0);
    mgr.record(InspectEvent::MemWrite, 0x2000, 4, 0);

    assert!(mgr.events().is_empty());
    assert_eq!(mgr.event_counts()[InspectEvent::MemWrite as usize], 2);
}

#[test]
fn test_inspection_on_state() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.inspection_mut().enable(InspectEvent::MemWrite);
    state.inspection_mut().enable(InspectEvent::MemRead);

    state.set_pc(0x400000);
    state.inspect_mem_write(0x1000, 8);
    state.inspect_mem_read(0x2000, 4);

    assert_eq!(
        state.inspection().event_counts()[InspectEvent::MemWrite as usize],
        1
    );
    assert_eq!(
        state.inspection().event_counts()[InspectEvent::MemRead as usize],
        1
    );
}

#[test]
fn test_inspection_fork_isolation() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.inspection_mut().enable(InspectEvent::MemWrite);
    state.inspect_mem_write(0x1000, 4);

    let mut forked = state.fork();
    forked.inspect_mem_write(0x2000, 4);

    // Parent should have 1 event
    assert_eq!(
        state.inspection().event_counts()[InspectEvent::MemWrite as usize],
        1
    );
    // Forked should have 2 (inherited 1 + new 1)
    assert_eq!(
        forked.inspection().event_counts()[InspectEvent::MemWrite as usize],
        2
    );
}

#[test]
fn test_inspection_disabled_no_record() {
    let mut state = RustSimState::new("amd64").unwrap();
    // Don't enable anything
    state.inspect_mem_write(0x1000, 4);
    state.inspect_mem_read(0x2000, 4);

    assert_eq!(state.inspection().events().len(), 0);
    assert_eq!(
        state.inspection().event_counts()[InspectEvent::MemWrite as usize],
        0
    );
}

// =========================================================================
// Snapshot / Serialization tests (angr-x04s.1.3)
// =========================================================================

/// Build a representative RustSimState that touches each bucket A/B/C
/// field (concrete + symbolic registers, mapped memory pages, solver
/// constraints, history, call stack, file system, hooks, environment,
/// flags). The Z3 ASTs are minted inside whichever Z3 context the test
/// is currently running under.
#[cfg(feature = "vex-engine-z3")]
fn build_populated_state() -> RustSimState {
    let mut s = RustSimState::new("amd64").unwrap();
    s.set_pc(0x4012a0);

    // Bucket B (registers): one concrete, one symbolic.
    s.set_register("rax", RustBV::concrete(0xdead_beef, 64));
    let rbx_sym = {
        let ctx = s.solver().borrow();
        RustBV::symbolic(&ctx, "rbx_sym", 64)
    };
    s.set_register("rbx", rbx_sym.clone());

    // Bucket B (memory): map a page with concrete bytes.
    s.memory_mut()
        .map_data(0x10_0000u64, &[1u8, 2, 3, 4, 5], Permission::RWX);

    // Bucket A: history + call stack.
    s.add_to_history(0x4011a0);
    s.add_to_history(0x4012a0);
    s.push_call(0x4012a0, 0x401400, 0x4012a5, 0x7fff_ffff_0000);
    // Heap metadata via the public heap_alloc/heap_free helpers.
    let a1 = s.heap_alloc(32);
    let _a2 = s.heap_alloc(64);
    let _ = s.heap_free(a1);

    // Bucket C: hook + environment + stdin symbols.
    s.add_hook(0x401200);
    s.setenv(b"PATH".to_vec(), b"/usr/bin".to_vec());
    s.setenv(b"HOME".to_vec(), b"/root".to_vec());
    s.record_stdin_symbol("stdin_chunk_0".to_string(), 16);

    // Flags.
    s.set_no_ip_concretization(true);
    s.set_keep_ip_symbolic(false);
    s.set_no_symbolic_jump_resolution(true);
    s.set_posix_brk(0x1B0_4000);
    s.set_mmap_base(0xC100_8000);
    s.set_tsc_counter(0x0010_000F_A000);
    s.set_getopt_cursor(7, 3);
    s.set_getopt_extern(GetoptExternAddrs {
        optind: Some(0x602000),
        optarg: Some(0x602008),
        optopt: Some(0x602010),
    });
    s.push_native_resume_frame(NativeResumeFrame {
        proc_name: "pthread_once".to_string(),
        resume_tag: 1,
        saved_args: vec![RustBV::concrete(0x602000, 64)],
        caller_return_addr: 0x4007a0,
    });

    // Solver constraints — `rbx > 10` must hold after restore.
    let cmp = {
        let ctx = s.solver().borrow();
        let ten = RustBV::concrete(10, 64);
        rbx_sym.ugt(&ten, &ctx)
    };
    s.solver().borrow().assume_true(&cmp);

    s
}

/// Round-trip assertions: bucket A scalars + bucket B field counts +
/// bucket C collection contents must match. Solver SAT + concretize
/// proves the assumed_constraints were faithfully replayed (the
/// SymContextSnapshot path is already covered separately, this just
/// confirms the wiring through RustSimStateSnapshot).
#[cfg(feature = "vex-engine-z3")]
fn assert_state_round_trip(orig: &RustSimState, restored: &RustSimState) {
    // Bucket A scalars.
    assert_eq!(restored.pc(), orig.pc());
    assert_eq!(restored.state_id(), orig.state_id());
    assert_eq!(restored.parent_id(), orig.parent_id());
    assert_eq!(
        restored.history().iter().copied().collect::<Vec<_>>(),
        orig.history().iter().copied().collect::<Vec<_>>()
    );
    assert_eq!(
        restored.detailed_history().len(),
        orig.detailed_history().len()
    );
    assert_eq!(restored.heap_brk(), orig.heap_brk());
    assert_eq!(restored.posix_brk(), orig.posix_brk());
    assert_eq!(restored.mmap_base(), orig.mmap_base());
    assert_eq!(restored.tsc_counter(), orig.tsc_counter());
    assert_eq!(restored.getopt_cursor(), orig.getopt_cursor());
    {
        let (ro, oo) = (restored.getopt_extern(), orig.getopt_extern());
        assert_eq!(ro.optind, oo.optind);
        assert_eq!(ro.optarg, oo.optarg);
        assert_eq!(ro.optopt, oo.optopt);
    }
    {
        let (rs, os) = (restored.native_resume_stack(), orig.native_resume_stack());
        assert_eq!(rs.len(), os.len());
        for (rf, of) in rs.iter().zip(os.iter()) {
            assert_eq!(rf.proc_name, of.proc_name);
            assert_eq!(rf.resume_tag, of.resume_tag);
            assert_eq!(rf.saved_args.len(), of.saved_args.len());
            assert_eq!(rf.caller_return_addr, of.caller_return_addr);
        }
    }
    assert_eq!(restored.no_ip_concretization(), orig.no_ip_concretization());
    assert_eq!(restored.keep_ip_symbolic(), orig.keep_ip_symbolic());
    assert_eq!(
        restored.no_symbolic_jump_resolution(),
        orig.no_symbolic_jump_resolution()
    );
    assert_eq!(restored.call_stack().len(), orig.call_stack().len());
    assert_eq!(restored.vex_arch(), orig.vex_arch());
    assert_eq!(restored.arch().name(), orig.arch().name());

    // Bucket A subset: heap metadata.
    assert_eq!(
        restored.heap_metadata().alloc_count(),
        orig.heap_metadata().alloc_count()
    );
    assert_eq!(
        restored.heap_metadata().free_count(),
        orig.heap_metadata().free_count()
    );

    // Bucket A: stdin symbols.
    assert_eq!(restored.stdin_symbols(), orig.stdin_symbols());

    // Bucket B: registers — concrete rax and symbolic rbx width.
    assert_eq!(
        restored.get_register("rax").and_then(|bv| bv.as_u64()),
        Some(0xdead_beef)
    );
    let restored_rbx = restored.get_register("rbx").expect("rbx present");
    assert_eq!(restored_rbx.width(), 64);

    // Bucket B: memory — first concrete bytes survived.
    let restored_bytes = restored
        .memory()
        .read_concrete_bytes_for_lift(crate::memory::Address::new(0x10_0000), 5)
        .expect("memory readable");
    assert_eq!(restored_bytes, vec![1, 2, 3, 4, 5]);

    // Bucket C: hooks + env.
    assert!(restored.is_hooked(0x401200));
    assert_eq!(
        restored.getenv(b"PATH").map(<[u8]>::to_vec),
        Some(b"/usr/bin".to_vec())
    );
    assert_eq!(
        restored.getenv(b"HOME").map(<[u8]>::to_vec),
        Some(b"/root".to_vec())
    );

    // Solver replayed — restored context must be SAT with the rbx > 10
    // constraint honored.
    assert!(restored.solver().borrow().is_sat());
    let restored_rbx_val = restored
        .solver()
        .borrow()
        .eval(&restored_rbx)
        .expect("rbx evaluable");
    assert!(
        restored_rbx_val > 10,
        "constraint rbx > 10 not honored after restore (got {restored_rbx_val})"
    );
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_state_snapshot_round_trip_buckets_a_b_c() {
    let mut orig = build_populated_state();
    let snap = orig.to_snapshot();
    let restored = RustSimState::from_snapshot(snap).expect("from_snapshot");
    assert_state_round_trip(&orig, &restored);
}

/// `state-id-never-reused` across a snapshot boundary: a restored state carries
/// an ID minted by a foreign counter, so `from_snapshot` must lift the local
/// counter above it. Otherwise a resumed manager re-mints IDs that are live in
/// its own stashes (angr-op0dn.13.14).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_from_snapshot_reserves_foreign_state_id() {
    let mut snap = build_populated_state().to_snapshot();
    // An ID far above anything this process's counter has issued.
    snap.state_id = u32::MAX as u64;
    let restored = RustSimState::from_snapshot(snap).expect("from_snapshot");
    assert_eq!(restored.state_id(), u32::MAX as u64);

    let fresh = build_populated_state();
    assert!(
        fresh.state_id() > restored.state_id(),
        "counter must be lifted past a restored ID; fresh={} restored={}",
        fresh.state_id(),
        restored.state_id(),
    );
    let forked = restored.fork_true(&RustBV::concrete(1, 1));
    assert!(
        forked.state_id() > restored.state_id(),
        "fork of a restored state must not re-mint a live ID; child={} parent={}",
        forked.state_id(),
        restored.state_id(),
    );
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_state_to_from_serialized_round_trip() {
    let mut orig = build_populated_state();
    let bytes = orig.to_serialized();
    assert_eq!(
        bytes[0], SNAPSHOT_VERSION,
        "envelope must carry version byte"
    );
    let restored = RustSimState::from_serialized(&bytes).expect("from_serialized");
    assert_state_round_trip(&orig, &restored);
}

#[test]
fn test_state_from_serialized_empty_envelope() {
    match RustSimState::from_serialized(&[]) {
        Err(SnapshotError::EmptyEnvelope) => {}
        Err(other) => panic!("expected EmptyEnvelope, got {other:?}"),
        Ok(_) => panic!("empty envelope must fail"),
    }
}

#[test]
fn test_state_from_serialized_version_mismatch() {
    // Bump-byte trick: build a real envelope, replace version byte, expect
    // a fast VersionMismatch.
    let mut orig = RustSimState::new("amd64").unwrap();
    let mut bytes = orig.to_serialized();
    bytes[0] = SNAPSHOT_VERSION.wrapping_add(1);
    match RustSimState::from_serialized(&bytes) {
        Err(SnapshotError::VersionMismatch { found, expected }) => {
            assert_eq!(expected, SNAPSHOT_VERSION);
            assert_eq!(found, SNAPSHOT_VERSION.wrapping_add(1));
        }
        Err(other) => panic!("expected VersionMismatch, got {other:?}"),
        Ok(_) => panic!("bad version must fail"),
    }
}

// angr-ahypj: RustSimState::translate_state cross-context whole-state twin.
// Validates that translate_into composes correctly over a real state — a
// symbolic register pinned by a path constraint must re-evaluate to the same
// concrete witness once both the register overlay and the constraint have been
// Z3_translate'd into a fresh target context. Exercises shared-AST identity
// (the `rax` leaf in the register and inside the constraint must hash-cons to
// the same node in the target context) and identity preservation.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_translate_state_cross_context() {
    use z3::{Config, Context};

    let mut state = RustSimState::new("amd64").unwrap();
    let x = {
        let s = state.solver().borrow();
        RustBV::symbolic(&s, "ahypj_rax", 64)
    };
    state.set_register("rax", x.clone());
    let constraint = {
        let s = state.solver().borrow();
        x.eq(&RustBV::concrete(0xdead_beef, 64), &s)
    };
    state.add_constraint(constraint);

    // angr-0xyq2 Fix 6: the filesystem carries context-bound RustBVs too
    // (fd content_sym + the file_contents registry) — pin a symbolic file
    // byte via a path constraint so a missed translate surfaces as a
    // foreign-context AST when the target-context solver evals it.
    let fbyte = {
        let s = state.solver().borrow();
        RustBV::symbolic(&s, "ahypj_file_byte", 8)
    };
    let fconstraint = {
        let s = state.solver().borrow();
        fbyte.eq(&RustBV::concrete(0x41, 8), &s)
    };
    state.add_constraint(fconstraint);
    state
        .file_system()
        .register_file_content("/tmp/ahypj", vec![fbyte, RustBV::concrete(0x42, 8)]);
    let fs_fd = state
        .file_system()
        .open("/tmp/ahypj".to_string(), FdFlags::ReadOnly);

    let original = Context::thread_local();
    let cfg = Config::new();
    let target = Context::new(&cfg);
    assert_ne!(
        original.get_z3_context().as_ptr() as usize,
        target.get_z3_context().as_ptr() as usize,
        "test bug: target == source context",
    );

    // translate_state asserts constraints into the new context's solver, which
    // builds against the thread-local — swap first (the target-worker model).
    Context::set_thread_local(&target);
    let translated = state.translate_state(&target);
    let rax_t = translated.get_register("rax").expect("rax present");
    let got = translated.solver().borrow().eval(&rax_t);
    let sat = translated.solver().borrow().is_sat();
    // Fix 6: the translated state's fs content BVs must live in the target
    // context — eval them through the target-context solver (a plain-cloned
    // foreign-context AST would misbehave here), mirroring the rax check.
    let content_t = translated
        .file_system_ref()
        .fd_content_sym(fs_fd)
        .expect("content_sym survives translate_state");
    let fd_byte = translated.solver().borrow().eval(&content_t[0]);
    let fd_concrete = content_t[1].as_u64();
    let registry_t = translated
        .file_system_ref()
        .file_content_for_path("/tmp/ahypj")
        .expect("file_contents registry survives translate_state");
    let reg_byte = translated.solver().borrow().eval(&registry_t[0]);
    Context::set_thread_local(&original);

    assert_eq!(
        got,
        Some(0xdead_beef),
        "translated rax must resolve via the transferred constraint",
    );
    assert!(sat, "translated state's solver must remain SAT");
    assert_eq!(
        fd_byte,
        Some(0x41),
        "translated fd content_sym byte must re-prove its witness",
    );
    assert_eq!(
        fd_concrete,
        Some(0x42),
        "concrete content_sym byte clones verbatim",
    );
    assert_eq!(
        reg_byte,
        Some(0x41),
        "translated file_contents registry byte must re-prove its witness",
    );
    assert_eq!(
        translated.state_id(),
        state.state_id(),
        "translate_state preserves identity (same state_id)",
    );
}

// angr-5khjd: SymContext::translate_into must propagate the source
// context's timeout_ms/deterministic/use_shared_lineage_solver atomic flags
// onto the target context, mirroring fork()'s three-line propagation
// (SymContext::new() defaults all three, so a naive translate silently
// reverts a deterministic/non-default-timeout/shared-lineage-solver source
// to defaults). Sets all three to non-default values on the source state
// before calling translate_state and asserts the translated twin carries
// the same values.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_translate_state_propagates_lineage_flags() {
    use z3::{Config, Context};

    let state = RustSimState::new("amd64").unwrap();
    {
        let s = state.solver().borrow();
        s.set_timeout(12345);
        s.set_deterministic(true);
        s.set_use_shared_lineage_solver(true);
    }

    let original = Context::thread_local();
    let cfg = Config::new();
    let target = Context::new(&cfg);
    assert_ne!(
        original.get_z3_context().as_ptr() as usize,
        target.get_z3_context().as_ptr() as usize,
        "test bug: target == source context",
    );

    Context::set_thread_local(&target);
    let translated = state.translate_state(&target);
    let (timeout_ms, deterministic, shared_lineage_solver) = {
        let s = translated.solver().borrow();
        (
            s.timeout_ms(),
            s.is_deterministic(),
            s.use_shared_lineage_solver(),
        )
    };
    Context::set_thread_local(&original);

    assert_eq!(
        timeout_ms, 12345,
        "translate_state must propagate timeout_ms from the source context",
    );
    assert!(
        deterministic,
        "translate_state must propagate the deterministic flag from the source context",
    );
    assert!(
        shared_lineage_solver,
        "translate_state must propagate use_shared_lineage_solver from the source context",
    );
}

// angr-1ilq.2: translate_state must Z3_translate every RustBV in
// native_resume_stack.saved_args, not plain-clone the stack. A symbolic
// saved_arg cloned into a foreign worker's context is a dangling cross-context
// AST → UB once that worker touches it. This A->B->A' round-trip pushes a frame
// whose saved_args mix a symbolic arg (pinned by a path constraint) and a
// concrete arg, re-homes the whole state through a fresh context and back, and
// asserts the symbolic arg re-proves its witness via the transferred constraint
// at every hop (the concrete arg clones verbatim). Without the translate, the
// second hop reads a foreign-context AST.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_translate_state_resume_stack_cross_context() {
    use z3::{Config, Context};

    let mut state = RustSimState::new("amd64").unwrap();
    // Symbolic saved_arg pinned to a known witness via a path constraint.
    let x = {
        let s = state.solver().borrow();
        RustBV::symbolic(&s, "ilq2_saved_arg", 64)
    };
    let constraint = {
        let s = state.solver().borrow();
        x.eq(&RustBV::concrete(0xCAFE_F00D_DEAD_BEEF, 64), &s)
    };
    state.add_constraint(constraint);
    // Frame mixes a symbolic arg (must translate) and a concrete arg (clones
    // verbatim) — covers both arms of RustBV::translate_into.
    state.push_native_resume_frame(NativeResumeFrame {
        proc_name: "ilq2_proc".to_string(),
        resume_tag: 7,
        saved_args: vec![x.clone(), RustBV::concrete(0x602000, 64)],
        caller_return_addr: 0x400600,
    });

    let original = Context::thread_local();

    // Re-prove the frame survived translation into `twin`'s context.
    let verify = |twin: &RustSimState, label: &str| {
        assert!(
            twin.solver().borrow().is_sat(),
            "{label}: translated solver must remain SAT",
        );
        let stack = twin.native_resume_stack();
        assert_eq!(stack.len(), 1, "{label}: resume stack preserved");
        let frame = &stack[0];
        assert_eq!(frame.proc_name, "ilq2_proc", "{label}: proc_name preserved");
        assert_eq!(frame.resume_tag, 7, "{label}: resume_tag preserved");
        assert_eq!(
            frame.caller_return_addr, 0x400600,
            "{label}: caller_return_addr preserved",
        );
        assert_eq!(frame.saved_args.len(), 2, "{label}: saved_args count");
        assert_eq!(
            twin.solver().borrow().eval(&frame.saved_args[0]),
            Some(0xCAFE_F00D_DEAD_BEEF),
            "{label}: symbolic saved_arg must re-prove its witness",
        );
        assert_eq!(
            frame.saved_args[1].as_u64(),
            Some(0x602000),
            "{label}: concrete saved_arg clones verbatim",
        );
    };

    // A -> B: translate the whole state into a fresh target context.
    let cfg_b = Config::new();
    let ctx_b = Context::new(&cfg_b);
    assert_ne!(
        original.get_z3_context().as_ptr() as usize,
        ctx_b.get_z3_context().as_ptr() as usize,
        "test bug: target == source context",
    );
    Context::set_thread_local(&ctx_b);
    let twin_b = state.translate_state(&ctx_b);
    verify(&twin_b, "A->B");

    // B -> A': translate the twin BACK into another fresh context — the hop
    // that catches a half-translated (foreign-context) resume-stack AST.
    let cfg_a2 = Config::new();
    let ctx_a2 = Context::new(&cfg_a2);
    Context::set_thread_local(&ctx_a2);
    let twin_a2 = twin_b.translate_state(&ctx_a2);
    verify(&twin_a2, "B->A'");

    Context::set_thread_local(&original);
}

// angr-9pwjd: production-sized translate_state round-trip validation.
//
// The synthetic kill-gate above proves the mechanism on a one-leaf state.
// This test scales it to a state shaped like a real mid-run state — many
// distinct symbolic leaves spread across both the register file and a
// symbolic memory region, each pinned by its own path constraint — and
// drives a full A->B->A round-trip. The literal Z3-bound trio
// (fairlight/sokohashv2/angry-reverser) cannot be captured into a pure-Rust
// test (no in-Rust real-binary loader; the engine is driven from Python),
// so we reconstruct an equivalently-shaped multi-leaf constrained state and
// assert the falsifiable claim the bead cares about: EVERY path constraint
// re-checks equal after translation, through a fresh context and back again.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_translate_state_production_sized_roundtrip() {
    use z3::{Config, Context};

    const N_MEM_LEAVES: u64 = 64;
    const MEM_BASE: u64 = 0x10000;

    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(MEM_BASE, N_MEM_LEAVES * 8, Permission::RWX);

    // Pin a couple of registers to distinct symbolic leaves.
    let (rax, rbx) = {
        let s = state.solver().borrow();
        (
            RustBV::symbolic(&s, "pwjd_rax", 64),
            RustBV::symbolic(&s, "pwjd_rbx", 64),
        )
    };
    state.set_register("rax", rax.clone());
    state.set_register("rbx", rbx.clone());
    {
        let s = state.solver().borrow();
        let c_rax = rax.eq(&RustBV::concrete(0x1111_2222_3333_4444, 64), &s);
        let c_rbx = rbx.eq(&RustBV::concrete(0x5555_6666_7777_8888, 64), &s);
        drop(s);
        state.add_constraint(c_rax);
        state.add_constraint(c_rbx);
    }

    // Spread N distinct symbolic leaves across a symbolic memory region, each
    // pinned to a distinct witness. Mirrors a deep stdin/heap symbolic region.
    for i in 0..N_MEM_LEAVES {
        let leaf = {
            let s = state.solver().borrow();
            RustBV::symbolic(&s, format!("pwjd_mem_{i}"), 64)
        };
        state
            .memory_store(MEM_BASE + i * 8, leaf.clone())
            .expect("store leaf");
        let witness = 0xC0DE_0000_0000_0000u128 + i as u128;
        let s = state.solver().borrow();
        let c = leaf.eq(&RustBV::concrete(witness, 64), &s);
        drop(s);
        state.add_constraint(c);
    }

    // Capture the witnesses every constraint must re-prove after translation.
    let mut expected: Vec<(u64, u128)> = Vec::with_capacity(N_MEM_LEAVES as usize);
    for i in 0..N_MEM_LEAVES {
        let cell = state.memory_load(MEM_BASE + i * 8, 8).expect("load cell");
        let val = state.solver().borrow().eval(&cell).expect("cell evaluable");
        expected.push((MEM_BASE + i * 8, val));
    }
    let exp_rax = state
        .solver()
        .borrow()
        .eval(&state.get_register("rax").unwrap())
        .unwrap();
    let exp_rbx = state
        .solver()
        .borrow()
        .eval(&state.get_register("rbx").unwrap())
        .unwrap();

    // Helper: assert a translated twin re-proves every captured witness.
    let verify = |twin: &RustSimState, label: &str| {
        assert!(
            twin.solver().borrow().is_sat(),
            "{label}: translated solver must remain SAT",
        );
        for (addr, want) in &expected {
            let cell = twin.memory_load(*addr, 8).expect("load translated cell");
            let got = twin.solver().borrow().eval(&cell);
            assert_eq!(
                got,
                Some(*want),
                "{label}: mem leaf @{addr:#x} must re-prove its witness",
            );
        }
        assert_eq!(
            twin.solver()
                .borrow()
                .eval(&twin.get_register("rax").unwrap()),
            Some(exp_rax),
            "{label}: rax must re-prove its witness",
        );
        assert_eq!(
            twin.solver()
                .borrow()
                .eval(&twin.get_register("rbx").unwrap()),
            Some(exp_rbx),
            "{label}: rbx must re-prove its witness",
        );
        assert_eq!(
            twin.state_id(),
            state.state_id(),
            "{label}: translate_state preserves identity",
        );
    };

    let original = Context::thread_local();

    // A -> B: translate the whole state into a fresh target context.
    let cfg_b = Config::new();
    let ctx_b = Context::new(&cfg_b);
    Context::set_thread_local(&ctx_b);
    let twin_b = state.translate_state(&ctx_b);
    verify(&twin_b, "A->B");

    // B -> A': translate the twin BACK into another fresh context. This is
    // the round-trip that caught the iter15 add_constraint log bug — if the
    // translated solver did not seed its z3_assertions log, the second hop
    // would see zero constraints and the witnesses would not re-prove.
    let cfg_a2 = Config::new();
    let ctx_a2 = Context::new(&cfg_a2);
    Context::set_thread_local(&ctx_a2);
    let twin_a2 = twin_b.translate_state(&ctx_a2);
    verify(&twin_a2, "B->A'");

    Context::set_thread_local(&original);
}

// angr-1ilq.1: the SAFE cross-worker migration twin of
// `test_translate_state_cross_context`. Instead of `translate_state` (which
// reads the *source* context cross-thread — unsound under work-stealing,
// hazard C), migration serializes the state to context-free bytes on the
// owner and rebuilds the ASTs in the stealing worker's own context. The same
// falsifiable claim must hold: a symbolic register pinned by a path constraint
// re-evaluates to its witness after the round-trip through a fresh context.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_migrate_via_snapshot_cross_context() {
    use z3::{Config, Context};

    let mut state = RustSimState::new("amd64").unwrap();
    let x = {
        let s = state.solver().borrow();
        RustBV::symbolic(&s, "ilq1_rax", 64)
    };
    state.set_register("rax", x.clone());
    let constraint = {
        let s = state.solver().borrow();
        x.eq(&RustBV::concrete(0xdead_beef, 64), &s)
    };
    state.add_constraint(constraint);
    let sid = state.state_id();

    let original = Context::thread_local();
    let target = Context::new(&Config::new());
    assert_ne!(
        original.get_z3_context().as_ptr() as usize,
        target.get_z3_context().as_ptr() as usize,
        "test bug: target == source context",
    );

    // Owner serializes under the source context (current thread-local)...
    let payload = state.detach_for_migration();
    // ...stealer swaps its context in, then rebuilds — every AST is minted in
    // `target`, with no read of the source context.
    Context::set_thread_local(&target);
    let migrated = payload.reattach(&target).expect("reattach");
    let rax_t = migrated.get_register("rax").expect("rax present");
    let got = migrated.solver().borrow().eval(&rax_t);
    let sat = migrated.solver().borrow().is_sat();
    Context::set_thread_local(&original);

    assert_eq!(
        got,
        Some(0xdead_beef),
        "migrated rax must resolve via the transferred constraint",
    );
    assert!(sat, "migrated state's solver must remain SAT");
    assert_eq!(
        migrated.state_id(),
        sid,
        "migration preserves identity (same state_id)",
    );
}

// angr-sqfj8.71: `detach_for_migration` (state/migration.rs) must flush
// Multi cells before serializing, else bytes installed by the lazy
// symbolic-address store path (`store_symbolic_unified` resolving to
// `Multiple`) are silently absent on the stealing worker — the exact
// "silently losing data on state migration" shape the bug describes.
// `RustSimState::to_snapshot`/`to_serialized` now flush internally
// (`flush_memory`), so this is a compiler-enforced invariant
// (`SymbolicMemory::to_snapshot` requires a `MultiFlushed` proof), not just a
// call-site convention; this test proves the DATA survives the round trip.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_migrate_carries_multi_cell_bytes() {
    use z3::{Config, Context};

    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x1000, 0x4000, crate::memory::Permission::RWX);

    let (addr_var, val_bv) = {
        let ctx = state.solver().borrow();
        let addr_var = RustBV::symbolic(&ctx, "sqfj8_71_multi_addr", 64);
        ctx.assume_true(
            &addr_var
                .eq(&RustBV::concrete(0x1000, 64), &ctx)
                .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
        );
        (addr_var, RustBV::concrete(0xCAFE_BABEu128, 32))
    };
    let concretizer = crate::concretize::AddressConcretizer {
        symbolic_write_addresses: true,
        ..crate::concretize::AddressConcretizer::new()
    };
    let ctx = state.solver().clone();
    let ctx = ctx.borrow();
    state
        .memory_mut()
        .store_symbolic_unified(addr_var, val_bv, &ctx, &concretizer)
        .expect("Multi store must succeed");
    drop(ctx);
    assert_ne!(
        state.memory().multi_cell_count(),
        0,
        "precondition: the store must have installed Multi cells"
    );

    let target = Context::new(&Config::new());
    let payload = state.detach_for_migration();
    Context::set_thread_local(&target);
    let migrated = payload.reattach(&target).expect("reattach");

    let loaded = migrated
        .memory()
        .get_symbolic_object(0x1000)
        .expect("byte 0x1000 must be a symbolic object after migration, not silently dropped");
    assert_eq!(
        loaded.width(),
        32,
        "migrated Multi-cell byte must round-trip at its stored width"
    );
}

// angr-1ilq.1: migration parity with
// `test_translate_state_resume_stack_cross_context` over an A->B->A' round
// trip. Proves the snapshot transport carries `native_resume_stack` (the
// angr-1ilq.2 field) faithfully across two distinct contexts: a symbolic
// saved_arg re-proves its witness at every hop; a concrete saved_arg clones
// verbatim. Each hop serializes under the context that is then live, so a
// half-rebuilt foreign-context AST would surface here.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_migrate_resume_stack_roundtrip() {
    use z3::{Config, Context};

    let mut state = RustSimState::new("amd64").unwrap();
    let x = {
        let s = state.solver().borrow();
        RustBV::symbolic(&s, "ilq1_saved_arg", 64)
    };
    let constraint = {
        let s = state.solver().borrow();
        x.eq(&RustBV::concrete(0xCAFE_F00D_DEAD_BEEF, 64), &s)
    };
    state.add_constraint(constraint);
    state.push_native_resume_frame(NativeResumeFrame {
        proc_name: "ilq1_proc".to_string(),
        resume_tag: 7,
        saved_args: vec![x.clone(), RustBV::concrete(0x602000, 64)],
        caller_return_addr: 0x400600,
    });

    let original = Context::thread_local();

    let verify = |twin: &RustSimState, label: &str| {
        assert!(
            twin.solver().borrow().is_sat(),
            "{label}: migrated solver must remain SAT",
        );
        let stack = twin.native_resume_stack();
        assert_eq!(stack.len(), 1, "{label}: resume stack preserved");
        let frame = &stack[0];
        assert_eq!(frame.proc_name, "ilq1_proc", "{label}: proc_name preserved");
        assert_eq!(frame.resume_tag, 7, "{label}: resume_tag preserved");
        assert_eq!(
            frame.caller_return_addr, 0x400600,
            "{label}: caller_return_addr preserved",
        );
        assert_eq!(frame.saved_args.len(), 2, "{label}: saved_args count");
        assert_eq!(
            twin.solver().borrow().eval(&frame.saved_args[0]),
            Some(0xCAFE_F00D_DEAD_BEEF),
            "{label}: symbolic saved_arg must re-prove its witness",
        );
        assert_eq!(
            frame.saved_args[1].as_u64(),
            Some(0x602000),
            "{label}: concrete saved_arg clones verbatim",
        );
    };

    // A -> B: serialize under A, rebuild under a fresh B.
    let payload_b = state.detach_for_migration();
    let ctx_b = Context::new(&Config::new());
    assert_ne!(
        original.get_z3_context().as_ptr() as usize,
        ctx_b.get_z3_context().as_ptr() as usize,
        "test bug: target == source context",
    );
    Context::set_thread_local(&ctx_b);
    let twin_b = payload_b.reattach(&ctx_b).expect("reattach A->B");
    verify(&twin_b, "A->B");

    // B -> A': serialize the twin under B, rebuild under another fresh context.
    let payload_a2 = twin_b.detach_for_migration();
    let ctx_a2 = Context::new(&Config::new());
    Context::set_thread_local(&ctx_a2);
    let twin_a2 = payload_a2.reattach(&ctx_a2).expect("reattach B->A'");
    verify(&twin_a2, "B->A'");

    Context::set_thread_local(&original);
}

// angr-1ilq.1: the fidelity gap fix. `to_serialized` drops the Bucket-D
// `Py<PyAny>` overlays (symbolic_pages / hook_symbolic_memory / addr_to_ast),
// so a snapshot-only migration would silently lose Python-side symbolic state
// on callback-heavy workloads. Migration carries them as live `Send` handles.
// This proves they survive a cross-context round-trip alongside the Rust-side
// register/constraint state.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_migrate_preserves_py_overlays() {
    use pyo3::prelude::*;
    use z3::{Config, Context};

    Python::initialize();

    let mut state = RustSimState::new("amd64").unwrap();
    let x = {
        let s = state.solver().borrow();
        RustBV::symbolic(&s, "ilq1_overlay_rax", 64)
    };
    state.set_register("rax", x.clone());
    let constraint = {
        let s = state.solver().borrow();
        x.eq(&RustBV::concrete(0x1234_5678, 64), &s)
    };
    state.add_constraint(constraint);

    // Populate one entry in each Bucket-D overlay map.
    Python::attach(|py| {
        let mut pages: std::collections::HashMap<u64, Py<PyAny>> = std::collections::HashMap::new();
        pages.insert(0x1000, py.None());
        state.replace_symbolic_pages(pages);
        state.set_hook_symbolic_memory(0x2000, py.None(), 8);
        state.set_addr_to_ast(0x3000, py.None(), 4);
    });

    let original = Context::thread_local();
    let target = Context::new(&Config::new());

    let payload = state.detach_for_migration();
    Context::set_thread_local(&target);
    let migrated = payload.reattach(&target).expect("reattach");
    // Rust-side state survived too.
    let rax_t = migrated.get_register("rax").expect("rax present");
    let got = migrated.solver().borrow().eval(&rax_t);
    Context::set_thread_local(&original);

    assert_eq!(got, Some(0x1234_5678), "migrated rax re-proves witness");
    assert_eq!(
        migrated.symbolic_pages().len(),
        1,
        "symbolic_pages carried across migration (not dropped like a bare snapshot)",
    );
    assert!(
        migrated.symbolic_pages().contains_key(&0x1000),
        "symbolic_pages key preserved",
    );
    assert_eq!(
        migrated.hook_symbolic_memory().len(),
        1,
        "hook_symbolic_memory carried across migration",
    );
    assert!(
        migrated.hook_symbolic_memory().contains_key(&0x2000),
        "hook_symbolic_memory key preserved",
    );
    assert_eq!(
        migrated.addr_to_ast().len(),
        1,
        "addr_to_ast carried across migration",
    );
    assert!(
        migrated.addr_to_ast().contains_key(&0x3000),
        "addr_to_ast key preserved",
    );
}

// angr-1ilq.1: reattach must FAIL FAST (not silently rebuild in the wrong
// context) when `target_ctx` is not the active thread-local. `from_serialized`
// mints ASTs in the active thread-local, so a stale thread-local would bind the
// rebuilt state to the wrong context — Z3 UB on the next query. The guard is
// unconditional (returns ContextMismatch), so this holds in release too.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_migrate_reattach_wrong_context_errors() {
    use z3::{Config, Context};

    let state = RustSimState::new("amd64").unwrap();
    let payload = state.detach_for_migration();

    // `target` is a fresh context that is NOT the active thread-local (we never
    // call set_thread_local), so reattach must reject it rather than rebuild.
    let target = Context::new(&Config::new());
    match payload.reattach(&target) {
        Err(SnapshotError::ContextMismatch) => {}
        other => panic!(
            "reattach with a non-active target_ctx must return ContextMismatch, got {:?}",
            other.map(|s| s.state_id()),
        ),
    }
}

// angr-1ilq.1: the end-to-end point of the bead — the payload is genuinely
// moved to ANOTHER OS thread and rebuilt there. Proves the `Send` transport
// works across a real thread boundary (not just a thread-local swap on one
// thread): the worker thread installs its own Z3 context, reattaches, and the
// migrated state re-proves its pinned witness in that thread's context.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_migrate_across_real_thread() {
    use z3::{Config, Context};

    let mut state = RustSimState::new("amd64").unwrap();
    let x = {
        let s = state.solver().borrow();
        RustBV::symbolic(&s, "ilq1_thread_rax", 64)
    };
    state.set_register("rax", x.clone());
    let constraint = {
        let s = state.solver().borrow();
        x.eq(&RustBV::concrete(0x5151_5151, 64), &s)
    };
    state.add_constraint(constraint);

    // Serialize on this (owner) thread, then MOVE the payload to a worker.
    let payload = state.detach_for_migration();
    let (got, sat) = std::thread::spawn(move || {
        // The worker installs its OWN Z3 context as the thread-local.
        let worker_ctx = Context::new(&Config::new());
        Context::set_thread_local(&worker_ctx);
        let migrated = payload.reattach(&worker_ctx).expect("reattach on worker");
        let rax_t = migrated.get_register("rax").expect("rax present");
        let got = migrated.solver().borrow().eval(&rax_t);
        let sat = migrated.solver().borrow().is_sat();
        (got, sat)
    })
    .join()
    .expect("worker thread panicked");

    assert_eq!(
        got,
        Some(0x5151_5151),
        "migrated rax must re-prove its witness in the worker thread's context",
    );
    assert!(
        sat,
        "migrated state's solver must remain SAT on the worker thread"
    );
}

// angr-ph300.51: RustSimState::merge must union the three Python-AST overlay
// maps (not take only self's) and carry the furthest-advanced allocator
// watermarks — otherwise a branch-B symbolic overlay byte is silently lost and
// the merged state's next malloc can alias live allocations from B's ITE arm.
#[test]
fn test_merge_unions_other_only_overlays_and_takes_max_brk() {
    Python::initialize();

    let mut a = RustSimState::new("amd64").unwrap();
    let mut b = RustSimState::new("amd64").unwrap();

    // Self (a) owns an overlay at 0x1000; other (b) owns a *distinct* overlay at
    // 0x2000 plus a conflicting one at 0x1000. b also advances the allocators.
    Python::attach(|py| {
        a.set_hook_symbolic_memory(0x1000, py.None(), 8);
        b.set_hook_symbolic_memory(0x1000, py.None(), 4); // conflict -> keep self
        b.set_hook_symbolic_memory(0x2000, py.None(), 8); // other-only -> survive
    });
    a.set_heap_brk(0x10_0000);
    b.set_heap_brk(0x20_0000); // b allocated further
    a.set_posix_brk(0x30_0000);
    b.set_posix_brk(0x11_0000); // a's is higher here
    a.set_mmap_base(0x40_0000);
    b.set_mmap_base(0x50_0000);

    let m0 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "ph30051_m0", 1)
    };
    let m1 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "ph30051_m1", 1)
    };
    let merged = a.merge(&[&b], &[m0, m1]);

    // Other-only overlay survived; conflicting key kept self's size (8, not 4).
    assert_eq!(
        merged.hook_symbolic_memory().len(),
        2,
        "other-only overlay must survive the merge"
    );
    assert!(
        merged.hook_symbolic_memory().contains_key(&0x2000),
        "b's 0x2000 overlay must be present in the merged state"
    );
    assert_eq!(
        merged
            .hook_symbolic_memory()
            .get(&0x1000)
            .map(|(_, sz)| *sz),
        Some(8),
        "conflicting overlay must keep self's entry (size 8), not other's (4)"
    );

    // Allocator watermarks take the furthest-advanced value per field.
    assert_eq!(merged.heap_brk(), 0x20_0000, "heap_brk must be max(a, b)");
    assert_eq!(merged.posix_brk(), 0x30_0000, "posix_brk must be max(a, b)");
    assert_eq!(merged.mmap_base(), 0x50_0000, "mmap_base must be max(a, b)");
}

// angr-9ke6b.173: the simulated RDTSC counter moved from a process-wide
// AtomicU64 onto RustSimState. It must default to TSC_INITIAL, survive a fork
// (a child continues the parent's clock and then diverges independently), and
// merge as a max watermark so the merged state's next RDTSC can't read earlier
// than a value one of the arms already observed.
#[test]
fn test_tsc_counter_forks_and_merges_as_watermark() {
    Python::initialize();

    let mut a = RustSimState::new("amd64").unwrap();
    assert_eq!(
        a.tsc_counter(),
        crate::vex::dirty::TSC_INITIAL,
        "a fresh state starts at the fixed initial TSC, not wherever some \
         other state left a global counter"
    );

    a.set_tsc_counter(crate::vex::dirty::TSC_INITIAL + 5 * crate::vex::dirty::TSC_STEP);
    let mut child = a.fork();
    assert_eq!(
        child.tsc_counter(),
        a.tsc_counter(),
        "fork inherits the parent's clock"
    );

    // Diverge: the child executes more RDTSCs than the parent.
    child.set_tsc_counter(child.tsc_counter() + 3 * crate::vex::dirty::TSC_STEP);
    assert_ne!(child.tsc_counter(), a.tsc_counter());

    let m0 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "tsc173_m0", 1)
    };
    let m1 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "tsc173_m1", 1)
    };
    let expected = child.tsc_counter();
    let merged = a.merge(&[&child], &[m0, m1]);
    assert_eq!(
        merged.tsc_counter(),
        expected,
        "merge must take the furthest-advanced TSC across branches"
    );
}

// angr-sqfj8.88: `last_time` is `Option<RustBV>`, the symbolic analogue of the
// `tsc_counter` watermark above — merge must never let it read as though time
// ran backwards relative to any branch. Since `RustBV` has no `Ord`, this
// takes a real symbolic maximum (`uge` + `ite`) rather than dropping straight
// to `self`'s value.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_merge_takes_symbolic_max_of_last_time() {
    Python::initialize();

    let mut a = RustSimState::new("amd64").unwrap();
    let mut b = a.fork();
    a.set_last_time(RustBV::concrete(100, 64));
    b.set_last_time(RustBV::concrete(250, 64));

    let m0 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "sqfj8_88_m0", 1)
    };
    let m1 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "sqfj8_88_m1", 1)
    };
    let merged = a.merge(&[&b], &[m0, m1]);

    let last = merged
        .last_time()
        .expect("last_time must survive the merge");
    let val = merged.solver().borrow().eval(last);
    assert_eq!(
        val,
        Some(250),
        "merge must take the later branch's time value, not silently keep \
         self's smaller one"
    );
}

/// angr-sqfj8.86: merge must actually detect a `native_resume_stack` length
/// mismatch across branches — before this fix nothing computed divergence
/// for this field at all, so `warn_config_divergence` never fired no matter
/// how far branches drifted. Exercises `native_resume_stack_diverges`
/// directly (the detection logic `RustSimState::merge` feeds into
/// `warn_config_divergence`) rather than trying to observe the resulting
/// `log::warn!` call: `log::set_logger` is a process-global one-shot
/// singleton, and `engine_tests.rs`'s `set_rust_log_level_accepts_levels_and_specs`
/// already claims that slot in the same test binary, so a competing logger
/// installed here would race it non-deterministically.
#[test]
fn test_native_resume_stack_diverges_detects_depth_mismatch() {
    let mut a = RustSimState::new("amd64").unwrap();
    let b = a.fork();
    assert!(
        !fork::native_resume_stack_diverges(&a, &[&b]),
        "equal-depth (both empty) stacks must not read as diverged"
    );

    a.push_native_resume_frame(NativeResumeFrame {
        proc_name: "pthread_once".to_string(),
        resume_tag: 0,
        saved_args: vec![RustBV::concrete(0x601000, 64)],
        caller_return_addr: 0x400600,
    });
    assert!(
        fork::native_resume_stack_diverges(&a, &[&b]),
        "a length-1 vs length-0 native_resume_stack must be detected as diverged"
    );

    // The merge itself still self-wins regardless of divergence detection
    // (see `RustSimState::merge`'s doc) — confirm that carry survives too.
    let m0 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "sqfj8_86_m0", 1)
    };
    let m1 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "sqfj8_86_m1", 1)
    };
    let merged = a.merge(&[&b], &[m0, m1]);
    assert_eq!(
        merged.native_resume_stack().len(),
        1,
        "merge must keep self's frame (self-wins), not silently b's empty stack"
    );
}

// angr-n0irt.2: RustSimState::merge must extend the angr-ph300.51 max-merge fix
// to the CGC allocator state. cgc_allocation_base grows DOWNWARD, so the anti-
// alias combinator is min (furthest-advanced base across branches), NOT the max
// used for the up-growing brk/mmap watermarks. cgc_sinkholes must INTERSECT (a
// freed region is only safe to reuse if it was freed in every branch — a region
// freed in one branch but live in another survives the page-unioning merge, so
// unioning sinkholes would reintroduce aliasing).
#[test]
fn test_merge_takes_min_cgc_base_and_intersects_sinkholes() {
    Python::initialize();

    let mut a = RustSimState::new("amd64").unwrap();
    let mut b = RustSimState::new("amd64").unwrap();
    let mut c = RustSimState::new("amd64").unwrap();

    // Downward-growing base: a is the default high-water, b and c bumped lower
    // (allocated more). min over all three is b's 0xB700_0000 — which is neither
    // self's value nor a plain "take other", so this distinguishes min from max
    // (max would be a's 0xB800_0000) and from "last other" (c's 0xB750_0000).
    a.set_cgc_allocation_base(0xB800_0000);
    b.set_cgc_allocation_base(0xB700_0000);
    c.set_cgc_allocation_base(0xB750_0000);

    // Only (0x1000, 0x100) is present in ALL three branches -> survives.
    a.cgc_add_sinkhole(0x1000, 0x100);
    a.cgc_add_sinkhole(0x2000, 0x100); // self-only -> dropped
    b.cgc_add_sinkhole(0x1000, 0x100);
    b.cgc_add_sinkhole(0x4000, 0x100); // other-only -> dropped
    c.cgc_add_sinkhole(0x1000, 0x100);
    c.cgc_add_sinkhole(0x5000, 0x100); // other-only -> dropped

    let m0 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "n0irt2_m0", 1)
    };
    let m1 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "n0irt2_m1", 1)
    };
    let m2 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "n0irt2_m2", 1)
    };
    let merged = a.merge(&[&b, &c], &[m0, m1, m2]);

    assert_eq!(
        merged.cgc_allocation_base(),
        0xB700_0000,
        "cgc_allocation_base must be min across all branches (grows downward)"
    );
    assert_eq!(
        merged.cgc_sinkholes(),
        &[(0x1000, 0x100)],
        "cgc_sinkholes must intersect: only the region freed in every branch survives"
    );
}

// angr-n0irt.3: RustSimState::merge must union heap_metadata across every
// branch, not keep only self's. heap_brk is maxed on merge, so a branch-only
// malloc stays reachable in the unioned memory — but if its alloc_size entry
// lived only on the dropped branch, NativeRealloc's copy length defaults to the
// FULL new size (old_size.map_or(size, ..)), over-reading past the true old
// allocation. Unioning allocations closes that; freed addresses union as a set.
#[test]
fn test_merge_unions_heap_metadata_across_branches() {
    Python::initialize();

    let mut a = RustSimState::new("amd64").unwrap();
    let mut b = RustSimState::new("amd64").unwrap();

    // self (a) allocates one block. other (b) allocates the same base block
    // (shared, as if inherited from a common fork point) plus a *second*,
    // higher block that lives only on its branch — that branch-only block must
    // survive the merge carrying its true size. a and b are independent states
    // starting from the same heap_brk, so b's first alloc aliases a's address.
    let a_addr = a.heap_alloc(0x10);
    let b_shared = b.heap_alloc(0x10);
    assert_eq!(a_addr, b_shared, "both start at the same heap_brk");
    let b_only = b.heap_alloc(0x40);
    assert_ne!(a_addr, b_only, "branch-only block sits at a higher address");

    // b frees the shared-address block only on its branch; the free must still
    // be recorded in the merged state for double-free / leak analysis.
    b.heap_free(b_shared);

    let (m0, m1) = {
        let s = a.solver().borrow();
        (
            RustBV::symbolic(&s, "n0irt3_m0", 1),
            RustBV::symbolic(&s, "n0irt3_m1", 1),
        )
    };
    let merged = a.merge(&[&b], &[m0, m1]);

    let hm = merged.heap_metadata();
    assert_eq!(
        hm.alloc_size(a_addr),
        Some(0x10),
        "self's allocation must survive the merge (a still holds this address)"
    );
    assert_eq!(
        hm.alloc_size(b_only),
        Some(0x40),
        "branch-only allocation must be unioned in with its true size, so \
         NativeRealloc copies min(new, 0x40) rather than over-reading"
    );
    assert!(
        hm.freed.contains(&b_shared),
        "other-branch free must be recorded in the merged state"
    );
}

// angr-n0irt.3 (unit): HeapMetadata::union_from keeps self's size on an
// address collision and unions `freed` as a set (no double-count of a free both
// branches inherited from a pre-fork allocation).
#[test]
fn test_heap_metadata_union_from() {
    let mut a = HeapMetadata::default();
    let mut b = HeapMetadata::default();

    a.record_alloc(0x1000, 0x10); // shared addr, self size wins
    a.record_alloc(0x2000, 0x20); // self-only
    b.record_alloc(0x1000, 0x99); // conflict -> dropped in favor of self's 0x10
    b.record_alloc(0x3000, 0x30); // other-only -> unioned in

    a.record_free(0x9000); // inherited free present on both branches
    b.record_free(0x9000); // dup -> not double-counted
    b.record_free(0x8000); // other-only free -> unioned in

    a.union_from(&b);

    assert_eq!(
        a.alloc_size(0x1000),
        Some(0x10),
        "self size wins on collision"
    );
    assert_eq!(
        a.alloc_size(0x2000),
        Some(0x20),
        "self-only allocation kept"
    );
    assert_eq!(
        a.alloc_size(0x3000),
        Some(0x30),
        "other-only allocation unioned"
    );
    assert_eq!(
        a.free_count(),
        2,
        "freed unioned as a set: {{0x9000, 0x8000}}"
    );
    assert!(a.freed.contains(&0x8000), "other-only free must be present");
}

// angr-sqfj8.85: `union_from` must not resurrect an address `self` already
// freed just because `other` never freed it on its branch — the check
// against `self.freed` has to gate the `or_insert`, not run after.
#[test]
fn test_heap_metadata_union_from_does_not_resurrect_freed_allocation() {
    let mut a = HeapMetadata::default();
    let mut b = HeapMetadata::default();

    a.record_alloc(0x4000, 0x10);
    a.record_free(0x4000); // self already freed this address
    b.record_alloc(0x4000, 0x10); // other's branch never freed it

    a.union_from(&b);

    assert_eq!(
        a.alloc_size(0x4000),
        None,
        "self's free must win — union must not resurrect a freed allocation"
    );
    assert!(
        a.freed.contains(&0x4000),
        "the address stays recorded as freed"
    );
}

// angr-9ke6b.127 (unit): `freed` is a set on the single-path side too — a
// double free of the same pointer records one entry, so `free_count` means the
// same thing whether the duplication arrived via `union_from` or via two
// `record_free` calls on one path.
#[test]
fn test_heap_metadata_record_free_dedups() {
    let mut hm = HeapMetadata::default();
    hm.record_alloc(0x1000, 0x10);
    hm.record_alloc(0x2000, 0x20);

    assert_eq!(
        hm.record_free(0x1000),
        Some(0x10),
        "first free returns size"
    );
    assert_eq!(
        hm.record_free(0x1000),
        None,
        "double free is already distinguishable by the None return"
    );
    hm.record_free(0x2000);
    hm.record_free(0x3000); // never allocated here; still recorded once
    hm.record_free(0x3000);

    assert_eq!(
        hm.free_count(),
        3,
        "freed is a set: {{0x1000, 0x2000, 0x3000}}"
    );
    assert_eq!(
        hm.freed,
        vec![0x1000, 0x2000, 0x3000],
        "first-free order preserved (export determinism)"
    );

    // A merge of a double-freeing path into another is idempotent too.
    let mut other = HeapMetadata::default();
    other.record_free(0x1000);
    other.record_free(0x4000);
    hm.union_from(&other);
    assert_eq!(hm.freed, vec![0x1000, 0x2000, 0x3000, 0x4000]);
}

// angr-ph300.75: RustSimState::merge keeps the longest-stdout branch's
// FileSystem wholesale (documented stdout-only merge contract). This locks in
// that contract: the merged fs is the longest-stdout branch's fd table, ties
// keep the earlier branch, and a dropped branch carrying non-stdout fds
// (file offsets/writes) is a detectable warn condition (its state is lost).
#[test]
fn test_merge_filesystem_keeps_longest_stdout_branch() {
    Python::initialize();

    let mut a = RustSimState::new("amd64").unwrap();
    let mut b = RustSimState::new("amd64").unwrap();

    // b writes more to stdout, so its FileSystem wins the merge. a opens a
    // non-std file fd and writes to it — that state is on the *dropped* branch.
    assert!(a.write_stdout(b"hi"));
    assert!(b.write_stdout(b"hello!"));
    let a_fd = a
        .file_system()
        .open("/tmp/only_on_a".to_string(), FdFlags::WriteOnly);
    assert!(a_fd > 2, "expected a non-std fd, got {a_fd}");
    assert!(a.write_fd(a_fd, b"branch-A-only"));

    // Precondition for the warn: a (the dropped branch) has fds above stderr,
    // b (the winner) does not.
    assert!(
        a.file_system_ref().has_fds_above_stderr(),
        "branch A must carry a non-std fd before merge"
    );
    assert!(
        !b.file_system_ref().has_fds_above_stderr(),
        "branch B must not carry a non-std fd"
    );

    let m0 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "ph30075_m0", 1)
    };
    let m1 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "ph30075_m1", 1)
    };
    let merged = a.merge(&[&b], &[m0, m1]);

    // The merged fs is b's (longest stdout): b's stdout survives and a's
    // non-std file fd is gone — the documented loss the warn announces.
    assert_eq!(
        merged.stdout_buffer(),
        b"hello!",
        "merged fs must be the longest-stdout branch (b)"
    );
    assert!(
        !merged.file_system_ref().has_fds_above_stderr(),
        "a's dropped non-std fd must NOT appear in the merged fs"
    );

    // Tie-break: equal stdout keeps self (a). Fresh states, both write the same
    // number of bytes, so a (self) wins and its non-std fd survives.
    let mut c = RustSimState::new("amd64").unwrap();
    let mut d = RustSimState::new("amd64").unwrap();
    assert!(c.write_stdout(b"eq"));
    assert!(d.write_stdout(b"eq"));
    let c_fd = c
        .file_system()
        .open("/tmp/only_on_c".to_string(), FdFlags::WriteOnly);
    assert!(c.write_fd(c_fd, b"branch-C-only"));
    let (n0, n1) = {
        let s = c.solver().borrow();
        (
            RustBV::symbolic(&s, "ph30075_n0", 1),
            RustBV::symbolic(&s, "ph30075_n1", 1),
        )
    };
    let merged_tie = c.merge(&[&d], &[n0, n1]);
    assert!(
        merged_tie.file_system_ref().has_fds_above_stderr(),
        "on a stdout tie the earlier branch (self=c) wins, keeping its non-std fd"
    );
}

// angr-qwyti.2 / angr-n0irt.18: systematic 3-state coverage of the overlay-union
// arms of RustSimState::merge (native/angr/src/state/fork.rs::merge). The
// pre-existing merge tests only exercised `hook_symbolic_memory` with 2 states,
// so two documented policies were asserted nowhere:
//   1. `symbolic_pages` and `addr_to_ast` share the identical union-with-
//      earlier-wins loop but were never touched by a test — a per-map typo
//      (e.g. inserting into the wrong map, or a `.contains_key`/`insert`
//      mismatch) would have gone undetected.
//   2. The 3-way tie-break "self > earlier-other > later-other": with only 2
//      states, a bug that iterated `others` in reverse (later-other winning a
//      conflict over earlier-other) passed silently. This needs >= 3 states
//      where the *same* key is present in two different `others` but NOT self.
//
// The test drives every overlay map through the same four-address table so a
// future map with the same policy can be slotted into the same asserts:
//   0x1000 present in all three  -> merged keeps self (a)          [self wins]
//   0x2000 present in b and c    -> merged keeps earlier-other (b) [tie-break]
//   0x3000 present in c only     -> survives                       [last-other]
//   0x4000 present in a only     -> survives                       [self-only]
// It also covers the `stdin_symbols` union arm (dedup by name, self-order-first),
// likewise untested before.
#[test]
fn test_merge_three_state_overlay_union_and_tiebreak() {
    Python::initialize();

    let mut a = RustSimState::new("amd64").unwrap();
    let mut b = RustSimState::new("amd64").unwrap();
    let mut c = RustSimState::new("amd64").unwrap();

    // Per-state, per-address entries. The *size* field (u32) of hook/addr_to_ast
    // entries and the *int value* of symbolic_pages ASTs double as a provenance
    // tag: 1 == came from a (self), 2 == b (earlier other), 3 == c (later other).
    Python::attach(|py| {
        let ast = |n: i64| n.into_pyobject(py).unwrap().into_any().unbind();

        // symbolic_pages: replace_symbolic_pages takes the whole map per state.
        let mut pa = std::collections::HashMap::new();
        pa.insert(0x1000u64, ast(1)); // all three -> self wins
        pa.insert(0x4000u64, ast(1)); // self-only -> survives
        a.replace_symbolic_pages(pa);
        let mut pb = std::collections::HashMap::new();
        pb.insert(0x1000u64, ast(2));
        pb.insert(0x2000u64, ast(2)); // b & c -> earlier-other (b) wins
        b.replace_symbolic_pages(pb);
        let mut pc = std::collections::HashMap::new();
        pc.insert(0x1000u64, ast(3));
        pc.insert(0x2000u64, ast(3));
        pc.insert(0x3000u64, ast(3)); // last-other-only -> survives
        c.replace_symbolic_pages(pc);

        // hook_symbolic_memory + addr_to_ast: same table, provenance in the size.
        for (st, tag) in [(&mut a, 1u32), (&mut b, 2), (&mut c, 3)] {
            st.set_hook_symbolic_memory(0x1000, py.None(), tag);
            st.set_addr_to_ast(0x1000, py.None(), tag);
        }
        for (st, tag) in [(&mut b, 2u32), (&mut c, 3)] {
            st.set_hook_symbolic_memory(0x2000, py.None(), tag);
            st.set_addr_to_ast(0x2000, py.None(), tag);
        }
        c.set_hook_symbolic_memory(0x3000, py.None(), 3);
        c.set_addr_to_ast(0x3000, py.None(), 3);
        a.set_hook_symbolic_memory(0x4000, py.None(), 1);
        a.set_addr_to_ast(0x4000, py.None(), 1);
    });

    // stdin_symbols union: b re-reads "sa" (dup of self) and adds "sb"; c adds
    // "sc". Merged must be self-order-first with dups dropped: [sa, sb, sc].
    a.record_stdin_symbol("sa".to_string(), 8);
    b.record_stdin_symbol("sa".to_string(), 8); // dup -> not re-added
    b.record_stdin_symbol("sb".to_string(), 8);
    c.record_stdin_symbol("sc".to_string(), 8);

    let (m0, m1, m2) = {
        let s = a.solver().borrow();
        (
            RustBV::symbolic(&s, "qwyti2_m0", 1),
            RustBV::symbolic(&s, "qwyti2_m1", 1),
            RustBV::symbolic(&s, "qwyti2_m2", 1),
        )
    };
    let merged = a.merge(&[&b, &c], &[m0, m1, m2]);

    // --- hook_symbolic_memory ---
    let hook = merged.hook_symbolic_memory();
    assert_eq!(hook.len(), 4, "hook: every distinct addr must survive");
    assert_eq!(
        hook.get(&0x1000).map(|(_, s)| *s),
        Some(1),
        "hook 0x1000: self wins"
    );
    assert_eq!(
        hook.get(&0x2000).map(|(_, s)| *s),
        Some(2),
        "hook 0x2000: earlier-other (b) wins the 3-way tie-break, not later-other (c)"
    );
    assert_eq!(
        hook.get(&0x3000).map(|(_, s)| *s),
        Some(3),
        "hook 0x3000: last-other survives"
    );
    assert_eq!(
        hook.get(&0x4000).map(|(_, s)| *s),
        Some(1),
        "hook 0x4000: self-only survives"
    );

    // --- addr_to_ast (identical policy, distinct map) ---
    let a2a = merged.addr_to_ast();
    assert_eq!(
        a2a.len(),
        4,
        "addr_to_ast: every distinct addr must survive"
    );
    assert_eq!(
        a2a.get(&0x1000).map(|(_, s)| *s),
        Some(1),
        "addr_to_ast 0x1000: self wins"
    );
    assert_eq!(
        a2a.get(&0x2000).map(|(_, s)| *s),
        Some(2),
        "addr_to_ast 0x2000: earlier-other (b) wins the tie-break"
    );
    assert_eq!(
        a2a.get(&0x3000).map(|(_, s)| *s),
        Some(3),
        "addr_to_ast 0x3000: last-other survives"
    );
    assert_eq!(
        a2a.get(&0x4000).map(|(_, s)| *s),
        Some(1),
        "addr_to_ast 0x4000: self-only survives"
    );

    // --- symbolic_pages (no size; provenance read from the int AST value) ---
    let pages = merged.symbolic_pages();
    assert_eq!(
        pages.len(),
        4,
        "symbolic_pages: every distinct addr must survive"
    );
    Python::attach(|py| {
        let tag = |addr: u64| pages.get(&addr).unwrap().bind(py).extract::<i64>().unwrap();
        assert_eq!(tag(0x1000), 1, "symbolic_pages 0x1000: self wins");
        assert_eq!(
            tag(0x2000),
            2,
            "symbolic_pages 0x2000: earlier-other (b) wins the tie-break"
        );
        assert_eq!(tag(0x3000), 3, "symbolic_pages 0x3000: last-other survives");
        assert_eq!(tag(0x4000), 1, "symbolic_pages 0x4000: self-only survives");
    });

    // --- stdin_symbols union (dedup by name, self-order-first) ---
    let names: Vec<&str> = merged
        .stdin_symbols()
        .iter()
        .map(|(n, _)| n.as_str())
        .collect();
    assert_eq!(
        names,
        vec!["sa", "sb", "sc"],
        "stdin_symbols must union across branches, dedup by name, self-order-first"
    );
}

/// angr-9ke6b.121 regression: the config-like fields a native proc or a Python
/// setter can mutate *after* a fork must not be carried from `self` alone.
/// `sim_options` / `hooks` / `environment` union; the four boolean SimOption
/// mirrors OR (an option a branch switched on stays on).
#[test]
fn test_merge_unions_config_like_fields_across_branches() {
    Python::initialize();

    let mut a = RustSimState::new("amd64").unwrap();
    let mut b = RustSimState::new("amd64").unwrap();

    // Options: one shared, one per branch.
    a.set_option("SHORT_READS", true);
    a.set_option("SELF_ONLY", true);
    b.set_option("SHORT_READS", true);
    b.set_option("OTHER_ONLY", true);
    // Boolean mirrors: only the *other* branch enables them, so a self-only
    // carry would silently revert both.
    b.set_keep_ip_symbolic(true);
    b.set_force_eager_forks(true);

    // Hooks: one per branch.
    a.add_hook(0x400000);
    b.add_hook(0x500000);

    // Environment: shared key with a conflicting value (self wins, warned) plus
    // a key only the other branch's setenv created.
    a.setenv(b"PATH".to_vec(), b"/a".to_vec());
    b.setenv(b"PATH".to_vec(), b"/b".to_vec());
    b.setenv(b"HOME".to_vec(), b"/root".to_vec());

    let (m0, m1) = {
        let s = a.solver().borrow();
        (
            RustBV::symbolic(&s, "ke6b121_m0", 1),
            RustBV::symbolic(&s, "ke6b121_m1", 1),
        )
    };
    let merged = a.merge(&[&b], &[m0, m1]);

    assert!(merged.has_option("SHORT_READS"), "shared option survives");
    assert!(merged.has_option("SELF_ONLY"), "self-only option survives");
    assert!(
        merged.has_option("OTHER_ONLY"),
        "other-branch-only option must be unioned in, not reverted to self's set"
    );
    assert!(
        merged.keep_ip_symbolic(),
        "keep_ip_symbolic mirrors a SimOption; a branch that enabled it wins"
    );
    assert!(
        merged.force_eager_forks(),
        "force_eager_forks mirrors a SimOption; a branch that enabled it wins"
    );

    assert!(merged.is_hooked(0x400000), "self's hook survives");
    assert!(
        merged.is_hooked(0x500000),
        "other-branch-only hook must be unioned in"
    );

    let env = merged.environment();
    assert_eq!(
        env.get(b"PATH".as_slice()).map(|v| v.as_slice()),
        Some(b"/a".as_slice()),
        "conflicting environment value keeps the earlier (self) branch's"
    );
    assert_eq!(
        env.get(b"HOME".as_slice()).map(|v| v.as_slice()),
        Some(b"/root".as_slice()),
        "other-branch-only environment key must be unioned in, not dropped"
    );
}

/// angr-9ke6b.121: the union must not allocate when nobody diverged — an
/// unmutated fork keeps sharing the parent's `Arc`s, so merging two pristine
/// forks leaves the sets pointer-identical to self's.
#[test]
fn test_merge_config_union_keeps_shared_arcs_when_nothing_diverged() {
    Python::initialize();

    let mut a = RustSimState::new("amd64").unwrap();
    a.set_option("SHORT_READS", true);
    a.add_hook(0x400000);
    a.setenv(b"PATH".to_vec(), b"/a".to_vec());
    let b = a.fork();

    let (m0, m1) = {
        let s = a.solver().borrow();
        (
            RustBV::symbolic(&s, "ke6b121b_m0", 1),
            RustBV::symbolic(&s, "ke6b121b_m1", 1),
        )
    };
    let merged = a.merge(&[&b], &[m0, m1]);

    assert!(merged.has_option("SHORT_READS"));
    assert!(merged.is_hooked(0x400000));
    assert_eq!(
        merged.environment().get(b"PATH".as_slice()).unwrap(),
        b"/a".as_slice()
    );
    assert!(
        !merged.keep_ip_symbolic() && !merged.force_eager_forks(),
        "OR must not invent a flag no branch set"
    );
}

/// Regression: a plain set union can't distinguish "never touched" from
/// "explicitly removed" — `a.remove_hook(X)` followed by `a.merge(&[&b])`
/// must NOT resurrect `X` just because sibling `b` never touched it. Same
/// bug class for `set_option(name, false)` and `unsetenv`. This is the
/// removal-direction counterpart to
/// `test_merge_unions_config_like_fields_across_branches` above, which only
/// exercised additions.
#[test]
fn test_merge_does_not_resurrect_removed_hook_option_or_env_var() {
    Python::initialize();

    let mut ancestor = RustSimState::new("amd64").unwrap();
    ancestor.add_hook(0x400000);
    ancestor.set_option("SHORT_READS", true);
    ancestor.setenv(b"PATH".to_vec(), b"/a".to_vec());

    let mut a = ancestor.fork();
    let b = ancestor.fork();

    // `a` explicitly removes everything; `b` never touches any of it.
    a.remove_hook(0x400000);
    a.set_option("SHORT_READS", false);
    a.unsetenv(b"PATH");

    let (m0, m1) = {
        let s = a.solver().borrow();
        (
            RustBV::symbolic(&s, "resurrect_m0", 1),
            RustBV::symbolic(&s, "resurrect_m1", 1),
        )
    };
    let merged = a.merge(&[&b], &[m0, m1]);

    assert!(
        !merged.is_hooked(0x400000),
        "a's explicit remove_hook must not be undone by b still having the hook"
    );
    assert!(
        !merged.has_option("SHORT_READS"),
        "a's explicit set_option(false) must not be undone by b still having it on"
    );
    assert!(
        merged.environment().get(b"PATH".as_slice()).is_none(),
        "a's explicit unsetenv must not be undone by b still having the var"
    );
}

/// Removal-then-merge must not regress the addition-direction fix
/// (`07f1f024e`): a hook/option/env-var added by `b` only, with `a` never
/// touching it, must still survive the merge.
#[test]
fn test_merge_still_unions_additions_alongside_removals() {
    Python::initialize();

    let mut ancestor = RustSimState::new("amd64").unwrap();
    ancestor.add_hook(0x400000);

    let mut a = ancestor.fork();
    let mut b = ancestor.fork();

    // `a` removes the shared hook; `b` independently adds a brand-new one.
    a.remove_hook(0x400000);
    b.add_hook(0x500000);
    b.set_option("OTHER_ONLY", true);
    b.setenv(b"HOME".to_vec(), b"/root".to_vec());

    let (m0, m1) = {
        let s = a.solver().borrow();
        (
            RustBV::symbolic(&s, "resurrect_add_m0", 1),
            RustBV::symbolic(&s, "resurrect_add_m1", 1),
        )
    };
    let merged = a.merge(&[&b], &[m0, m1]);

    assert!(
        !merged.is_hooked(0x400000),
        "a's removal still wins even when b also contributed an unrelated addition"
    );
    assert!(
        merged.is_hooked(0x500000),
        "b's brand-new hook must still be unioned in"
    );
    assert!(
        merged.has_option("OTHER_ONLY"),
        "b's brand-new option must still be unioned in"
    );
    assert_eq!(
        merged
            .environment()
            .get(b"HOME".as_slice())
            .map(|v| v.as_slice()),
        Some(b"/root".as_slice()),
        "b's brand-new env var must still be unioned in"
    );
}

/// A hook/option/env-var removed and then re-added on the SAME branch before
/// merge must survive: the tombstone has to clear on re-add, or the merge
/// would incorrectly treat it as still-removed and drop it out from under a
/// branch that currently has it live.
#[test]
fn test_merge_reinstated_hook_option_env_survives_own_removal_tombstone() {
    Python::initialize();

    let ancestor = RustSimState::new("amd64").unwrap();
    let mut a = ancestor.fork();
    let b = ancestor.fork();

    a.add_hook(0x400000);
    a.remove_hook(0x400000);
    a.add_hook(0x400000); // back on before merge — tombstone must clear.

    a.set_option("SHORT_READS", true);
    a.set_option("SHORT_READS", false);
    a.set_option("SHORT_READS", true);

    a.setenv(b"PATH".to_vec(), b"/a".to_vec());
    a.unsetenv(b"PATH");
    a.setenv(b"PATH".to_vec(), b"/a2".to_vec());

    let (m0, m1) = {
        let s = a.solver().borrow();
        (
            RustBV::symbolic(&s, "reinstate_m0", 1),
            RustBV::symbolic(&s, "reinstate_m1", 1),
        )
    };
    let merged = a.merge(&[&b], &[m0, m1]);

    assert!(
        merged.is_hooked(0x400000),
        "re-adding after remove_hook must clear the tombstone"
    );
    assert!(
        merged.has_option("SHORT_READS"),
        "re-enabling after set_option(false) must clear the tombstone"
    );
    assert_eq!(
        merged
            .environment()
            .get(b"PATH".as_slice())
            .map(|v| v.as_slice()),
        Some(b"/a2".as_slice()),
        "re-setenv after unsetenv must clear the tombstone"
    );
}

// =============================================================================
// export.rs — ExplorationStateSnapshot / RustSimState::export_full coverage
// (angr-n0irt.17). Before this, state/export.rs — the single choke point every
// found/errored/exported state crosses — had no #[cfg(test)] and no sibling
// *_tests.rs. These pin the zero-coverage branches named in the bead:
// the named-vs-symbolic register split (angr-4ju9e), memory_load page-boundary
// logic, get_page/page_addresses/get_symbolic_offsets, heap alloc/free, and
// open-fd export.
// =============================================================================

/// angr-4ju9e regression: export_full must (a) emit a *concrete* register in
/// `named_registers` (recoverable without an offset table on the Python side)
/// and (b) list a *symbolic* register in `symbolic_register_names` while
/// EXCLUDING it from `named_registers`, so Python attaches a lazy proxy instead
/// of silently dropping the Rust-computed symbolic value.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_export_full_named_vs_symbolic_register_split() {
    let mut s = RustSimState::new("amd64").unwrap();
    s.set_register("rax", RustBV::concrete(0xdead_beef, 64));
    let rbx_sym = {
        let ctx = s.solver().borrow();
        RustBV::symbolic(&ctx, "n0irt17_rbx", 64)
    };
    s.set_register("rbx", rbx_sym);

    let snap = s.export_full();

    let named = snap.get_registers_named();
    assert_eq!(
        named.get("rax"),
        Some(&(0xdead_beefu128, 64)),
        "concrete rax must be pre-computed into named_registers with its bit width"
    );
    assert!(
        !named.contains_key("rbx"),
        "symbolic rbx must be EXCLUDED from named_registers (else it is dropped)"
    );

    let sym_names = snap.get_symbolic_register_names();
    assert!(
        sym_names.iter().any(|n| n == "rbx"),
        "symbolic rbx must appear in symbolic_register_names for lazy-proxy recovery"
    );
    assert!(
        !sym_names.iter().any(|n| n == "rax"),
        "concrete rax must NOT be tagged symbolic"
    );
}

/// Memory export surface: page_count / page_addresses / get_page, and the
/// page-boundary offset+size arithmetic in `memory_load` (in-page hit, exact
/// end-of-page fit, one-byte overflow past the page → None, and unmapped →
/// None). Concrete-only pages report no symbolic offsets.
#[test]
fn test_export_full_memory_pages_and_load_boundary() {
    let mut s = RustSimState::new("amd64").unwrap();
    s.map_memory_data(0x10_0000, &[1u8, 2, 3, 4, 5, 6, 7, 8], Permission::RWX);
    s.map_memory_data(0x20_0000, &[0xAAu8; 4], Permission::RW);

    let snap = s.export_full();

    assert_eq!(snap.page_count(), 2, "two mapped pages must be exported");
    let addrs = snap.page_addresses();
    assert!(addrs.contains(&0x10_0000) && addrs.contains(&0x20_0000));

    // get_page returns a full PAGE_SIZE-wide, page-aligned tuple; OOB → None.
    let pg = snap.get_page(0).expect("page 0 exists");
    assert_eq!(pg.0 & 0xFFF, 0, "exported page addr must be page-aligned");
    assert_eq!(
        pg.1.len(),
        crate::memory::PAGE_SIZE as usize,
        "exported page must carry the whole page's concrete bytes"
    );
    assert!(snap.get_page(99).is_none(), "OOB page index → None");

    // In-page load at the mapped bytes.
    assert_eq!(
        snap.memory_load(0x10_0000, 4),
        Some(vec![1u8, 2, 3, 4]),
        "load of the seeded prefix must return those bytes"
    );
    // Exact end-of-page fit: offset 4092 + size 4 == PAGE_SIZE.
    assert_eq!(
        snap.memory_load(0x10_0000 + 4092, 4).map(|v| v.len()),
        Some(4),
        "a load that ends exactly at the page boundary must succeed"
    );
    // One byte past the page must NOT silently span into the next page.
    assert_eq!(
        snap.memory_load(0x10_0000 + 4093, 4),
        None,
        "a load overflowing the page boundary must return None"
    );
    // Unmapped address → None.
    assert_eq!(snap.memory_load(0x30_0000, 1), None);

    // Concrete-only page → no symbolic offsets; unknown page → empty too.
    assert!(snap.get_symbolic_offsets(0x10_0000).is_empty());
    assert!(snap.get_symbolic_offsets(0xDEAD_0000).is_empty());
}

/// A symbolic store marks the page's symbolic bitmap, and export_full must
/// surface those byte offsets via get_symbolic_offsets (the non-empty branch).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_export_full_symbolic_offsets_nonempty() {
    let mut s = RustSimState::new("amd64").unwrap();
    s.map_memory(0x40_0000, 0x1000, Permission::RW);
    let sym = {
        let ctx = s.solver().borrow();
        RustBV::symbolic(&ctx, "n0irt17_byte", 8)
    };
    s.memory_store(0x40_0000, sym).expect("symbolic store");

    let snap = s.export_full();
    assert_eq!(
        snap.get_symbolic_offsets(0x40_0000),
        vec![0u16],
        "byte 0 of the page must be flagged symbolic after an 8-bit symbolic store"
    );
}

/// Heap and open-fd export: a freed allocation moves out of `heap_allocated`
/// into `heap_freed`, and an open file descriptor is exported with its name,
/// content length, and open flag.
#[test]
fn test_export_full_heap_and_open_fds() {
    let mut s = RustSimState::new("amd64").unwrap();
    let a1 = s.heap_alloc(32);
    let a2 = s.heap_alloc(64);
    assert_eq!(s.heap_free(a1), Some(32), "free returns the tracked size");

    let fd =
        s.fs.open_with_content("/flag.txt".to_string(), FdFlags::ReadOnly, vec![b'A'; 10]);

    let snap = s.export_full();

    // a2 stays allocated; a1 has moved to freed.
    assert_eq!(snap.get_heap_alloc_count(), 1);
    assert_eq!(snap.get_heap_allocated(), vec![(a2, 64)]);
    assert_eq!(snap.get_heap_free_count(), 1);
    assert_eq!(snap.get_heap_freed(), vec![a1]);

    // The opened fd is exported with name, content length, and open flag.
    let fds = snap.get_open_fds();
    assert_eq!(
        snap.get_fd_count(),
        fds.len(),
        "fd_count must match the exported fd list length"
    );
    let entry = fds
        .iter()
        .find(|e| e.0 == fd)
        .expect("the opened fd must be exported");
    assert_eq!(entry.1, "/flag.txt", "fd name must round-trip");
    assert_eq!(entry.4, 10, "content length must be exported");
    assert!(entry.5, "a freshly opened fd must be exported as open");
}

/// Broad smoke over the remaining getters on a fully-populated state: raw
/// register bytes, history, call stack (list + depth), and arch name are all
/// exported non-trivially.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_export_full_smoke_history_callstack_arch() {
    let snap = build_populated_state().export_full();

    assert!(
        !snap.get_registers_raw().is_empty(),
        "raw register bytes must be exported"
    );
    assert!(!snap.arch_name.is_empty(), "arch name must be exported");
    assert!(
        !snap.get_history().is_empty(),
        "populated state has basic-block history"
    );
    assert_eq!(
        snap.get_call_stack_depth(),
        snap.get_call_stack().len(),
        "call_stack_depth must match the exported call-stack length"
    );
    assert!(
        snap.get_call_stack_depth() >= 1,
        "populated state pushed a call frame"
    );
}

/// angr-9ke6b.96: `memory_load` — the SimProcedure-facing load exposed as
/// the `memory_load` pymethod — goes through `SymbolicMemory::load_concrete`,
/// not the `_lazy` variant. A prior symbolic-address store that concretized
/// to Multiple/Strided leaves the value in un-flushed Multi cells with the
/// page's symbolic bitmap clear, so before the fix this returned the stale
/// concrete placeholder bytes with no error.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_memory_load_sees_unflushed_multi_cells() {
    let mut state = RustSimState::new("amd64").unwrap();
    // SYMBOLIC_WRITE_ADDRESSES on: without it an unannotated symbolic-address
    // store uses Python's Max-only chain and lands on one address, installing
    // no Multi cells (angr-9ke6b.194).
    state.set_concretizer(crate::concretize::AddressConcretizer {
        symbolic_write_addresses: true,
        ..crate::concretize::AddressConcretizer::new()
    });
    state.map_memory(0x1000, 0x4000, crate::memory::Permission::RWX);

    let addr_var = {
        let ctx = state.solver().borrow();
        let addr_var = RustBV::symbolic(&ctx, "state_multi_addr", 64);
        ctx.assume_true(
            &addr_var
                .eq(&RustBV::concrete(0x1000, 64), &ctx)
                .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
        );
        addr_var
    };

    state
        .memory_store_symbolic(addr_var.clone(), RustBV::concrete(0xDEADBEEF, 32))
        .expect("symbolic-address store must succeed");
    assert_eq!(
        state.memory().multi_cell_count(),
        8,
        "Multiple/Strided concretization installs per-byte Multi cells"
    );

    let loaded = state
        .memory_load(0x1000, 4)
        .expect("memory_load must succeed");
    assert!(
        loaded.as_u64().is_none(),
        "a Multi-covered load must be symbolic, not the placeholder concrete byte"
    );
    let probe = state.solver().borrow().fork();
    probe.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe));
    assert_eq!(
        probe.eval(&loaded),
        Some(0xDEADBEEF),
        "memory_load must reconstruct the Multi alternative"
    );
}

/// `apply_changes` splits a Python SimProcedure memory write into 16-byte
/// chunks (RustBV is u128-backed), so a buffer wider than one chunk must land
/// byte-for-byte rather than repeating the first 16 bytes.
#[test]
fn apply_changes_chunks_memory_writes_wider_than_16_bytes() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x1000, 0x1000, Permission::RWX);

    let data: Vec<u8> = (0..40u8).collect();
    let mut changes = StateChanges::new();
    changes.memory_writes.push((0x1000, data.clone()));
    state.apply_changes(&changes);

    for (i, &expected) in data.iter().enumerate() {
        let byte = state
            .memory_load(0x1000 + i as u64, 1)
            .expect("mapped load")
            .as_u64();
        assert_eq!(byte, Some(expected as u64), "byte {i} did not land");
    }
}

/// A write whose address is unmapped fails inside `store_concrete`.
/// `apply_changes` has no error channel to its callers, so it must warn and
/// keep applying the *remaining* changes instead of aborting the whole sync
/// (angr-sqfj8.87).
#[test]
fn apply_changes_continues_past_a_failed_memory_write() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x1000, 0x1000, Permission::RWX);

    let mut changes = StateChanges::new();
    // Unmapped — store_concrete returns MemoryError::Unmapped.
    changes.memory_writes.push((0xdead_0000, vec![0xAA; 4]));
    // Mapped — must still be applied after the failure above.
    changes
        .memory_writes
        .push((0x1000, vec![0x11, 0x22, 0x33, 0x44]));
    changes.new_pc = Some(0x2000);
    state.apply_changes(&changes);

    assert_eq!(
        state.memory_load(0x1000, 4).expect("mapped load").as_u64(),
        Some(0x4433_2211),
        "a later write must survive an earlier unmapped write"
    );
    assert_eq!(
        state.pc(),
        0x2000,
        "PC update must survive the failed write"
    );
}

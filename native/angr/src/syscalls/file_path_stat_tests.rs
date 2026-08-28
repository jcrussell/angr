//! `stat` tests — path-keyed lookup, per-arch statbuf layout, fallbacks.

use super::*;
use super::stat_layouts::{S_IFREG_0755, ST_BLKSIZE};
use super::file_path_tests_support::*;
use crate::memory::Permission;
use crate::state::{FdFlags, RustSimState};
use crate::symbolic::RustBV;
use crate::syscalls::NativeSyscall;

#[test]
fn stat_unknown_path_returns_minus_one() {
    let mut state = state_with_path(b"/tmp/never-registered");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeStatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("stat ok");
    assert_eq!(expect_continue(out), NEG_ONE);
    // Buffer must NOT have been touched on the failure path.
    assert_eq!(read_u64_le(&state, 0x4000), 0);
}

#[test]
fn stat_empty_path_returns_minus_one() {
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    state
        .memory_store(0x2000, RustBV::concrete(0, 8))
        .expect("nul");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeStatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("stat ok");
    assert_eq!(expect_continue(out), NEG_ONE);
}

#[test]
fn stat_known_path_with_content_writes_amd64_layout() {
    let mut state = state_with_path(b"/tmp/sized");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    // Seed an fd with 13 bytes so content_size_for_path returns Some(13).
    let _fd = state
        .file_system()
        .open_with_content(
            "/tmp/sized".into(),
            FdFlags::ReadOnly,
            b"hello, world!".to_vec(),
        )
        .expect("fd space is not exhausted in tests");

    let out = NativeStatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("stat ok");
    assert_eq!(expect_continue(out), 0);

    // st_size at offset 0x30
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 13);
    // st_mode at offset 0x18 (u32)
    assert_eq!(read_u32_le(&state, 0x4000 + 0x18), S_IFREG_0755 as u32);
    // st_blksize at offset 0x38
    assert_eq!(read_u64_le(&state, 0x4000 + 0x38), ST_BLKSIZE);
}

/// angr-0xyq2 Phase 1: `stat` on a path whose fd carries bounded symbolic
/// content reports the symbolic byte count via `content_size_for_path`
/// (which now keys off `effective_len`, not the empty concrete buffer).
#[test]
fn stat_symbolic_content_path_reports_symbolic_len() {
    let mut state = state_with_path(b"/tmp/symstat");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let bytes: Vec<RustBV> = (0..9u64)
        .map(|i| RustBV::symbolic_with_id(0x7f000 + i, format!("stat_sym_{i}"), 8))
        .collect();
    state
        .file_system()
        .register_file_content("/tmp/symstat", bytes);
    let _fd = state
        .file_system()
        .open("/tmp/symstat".into(), FdFlags::ReadOnly)
        .expect("fd space is not exhausted in tests");

    let out = NativeStatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("stat ok");
    assert_eq!(expect_continue(out), 0);

    // st_size at offset 0x30 — the symbolic byte count.
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 9);
}

#[test]
fn stat_registered_path_without_fd_uses_zero_size() {
    let mut state = state_with_path(b"/etc/registered-only");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    // Register without allocating an fd — content_size_for_path → None.
    state
        .file_system()
        .register_known_path("/etc/registered-only".to_string());

    let out = NativeStatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("stat ok");
    assert_eq!(expect_continue(out), 0);
    // st_size defaults to 0 when content_size_for_path is None.
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 0);
    assert_eq!(read_u32_le(&state, 0x4000 + 0x18), S_IFREG_0755 as u32);
}

/// Fix 1 (angr-0xyq2 review): `content_size_for_path` consults the
/// `file_contents` registry, so `stat` on a registered-but-never-opened
/// path reports the registered content length (not the zero-size default).
#[test]
fn stat_registered_content_path_without_fd_reports_len() {
    let mut state = state_with_path(b"/tmp/regonly");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let bytes: Vec<RustBV> = (0..7u64)
        .map(|i| RustBV::symbolic_with_id(0x8f000 + i, format!("regonly_{i}"), 8))
        .collect();
    state
        .file_system()
        .register_file_content("/tmp/regonly", bytes);
    // No open — the registry alone must supply the size.

    let out = NativeStatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("stat ok");
    assert_eq!(expect_continue(out), 0);
    // st_size at offset 0x30 — the registered byte count.
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 7);
}

/// Fix 2 failure A (angr-0xyq2 review): after `register_file_content` of a
/// relative path, `access` and `stat` with the same relative spelling must
/// succeed — known_paths stores the cwd-normalized key and `is_path_known`
/// / `content_size_for_path` normalize their queries.
#[test]
fn access_and_stat_relative_spelling_after_relative_registration() {
    let mut state = RustSimState::new("amd64").expect("state");
    state.file_system().set_cwd(b"/home/user".to_vec());
    let bytes: Vec<RustBV> = (0..5u64)
        .map(|i| RustBV::symbolic_with_id(0x9f000 + i, format!("relreg_{i}"), 8))
        .collect();
    state.file_system().register_file_content("flag.txt", bytes);
    stage_path(&mut state, 0x2000, b"flag.txt");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeAccessSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0, 64)],
        )
        .expect("access ok");
    assert_eq!(expect_continue(out), 0);

    let out = NativeStatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("stat ok");
    assert_eq!(expect_continue(out), 0);
    // st_size at offset 0x30 — the registered byte count.
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 5);
}

/// Fix 2 failure B (angr-0xyq2 review): after a relative `open` (the fd
/// stores the raw relative name), `stat` of the absolute spelling must
/// find the fd and report its size — `content_size_for_path` normalizes
/// the stored fd names on the fly for comparison.
#[test]
fn stat_absolute_spelling_after_relative_open_reports_size() {
    let mut state = state_with_path(b"/home/user/notes.txt");
    state.map_memory(0x4000, 0x1000, Permission::RW);
    state.file_system().set_cwd(b"/home/user".to_vec());

    let _fd = state
        .file_system()
        .open_with_content("notes.txt".into(), FdFlags::ReadOnly, b"hello".to_vec())
        .expect("fd space is not exhausted in tests");

    let out = NativeStatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("stat ok");
    assert_eq!(expect_continue(out), 0);
    // st_size at offset 0x30 — the relative fd's concrete content length.
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 5);
}

#[test]
fn stat_unsupported_arch_falls_back() {
    // ARM64 has no legacy stat (only newfstatat). MIPS64 carries the
    // legacy 64-bit struct stat with no native writer — handler errors
    // out so the dispatcher falls back to Python's error path. X86
    // (angr-11djq.5.1), ARM (angr-11djq.5.2) and MIPS32 (angr-11djq.5.3,
    // via the LFS stat64 number) ARE supported, so they are excluded here.
    for arch in ["aarch64", "mips64"] {
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();
        // Register the path so we'd otherwise succeed.
        state
            .file_system()
            .register_known_path("/tmp/foo".to_string());

        let err = NativeStatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, bits),
                    RustBV::concrete(0x4000, bits),
                ],
            )
            .expect_err("must surface as Other");
        match err {
            SyscallError::Other(msg) => {
                assert!(msg.contains("unsupported arch"), "{arch}: got {msg:?}",)
            }
            other => panic!("{arch}: expected Other, got {other:?}"),
        }
    }
}

#[test]
fn stat_symbolic_pathname_addr_falls_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    let sym_ptr = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "pathname_ptr", 64)
    };
    let err = NativeStatSyscall
        .call(&mut state, &[sym_ptr, RustBV::concrete(0x4000, 64)])
        .expect_err("must fall back");
    assert_symbolic_arg(err, "pathname");
}

#[test]
fn stat_symbolic_buf_falls_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    let sym_buf = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "statbuf", 64)
    };
    let err = NativeStatSyscall
        .call(&mut state, &[RustBV::concrete(0x2000, 64), sym_buf])
        .expect_err("must fall back");
    assert_symbolic_arg(err, "statbuf");
}

#[test]
fn stat_unmapped_buf_surfaces_error() {
    let mut state = state_with_path(b"/tmp/known");
    state
        .file_system()
        .register_known_path("/tmp/known".to_string());
    // Do NOT map the statbuf page.
    let err = NativeStatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x8000, 64)],
        )
        .expect_err("unmapped should error");
    assert_memory_err(err);
}

#[test]
fn stat_uses_largest_content_len_across_fds_for_same_path() {
    let mut state = state_with_path(b"/tmp/shared");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    // Two fds for the same name with different sizes. Helper takes
    // the max (deterministic regardless of iteration order).
    let _fd_small = state
        .file_system()
        .open_with_content("/tmp/shared".into(), FdFlags::ReadOnly, vec![0u8; 4])
        .expect("fd space is not exhausted in tests");
    let _fd_big = state
        .file_system()
        .open_with_content("/tmp/shared".into(), FdFlags::ReadOnly, vec![0u8; 17])
        .expect("fd space is not exhausted in tests");

    let out = NativeStatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("stat ok");
    assert_eq!(expect_continue(out), 0);
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 17);
}

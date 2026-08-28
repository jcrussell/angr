//! `lstat` tests (angr-poao).
//!
//! lstat is a stat() clone for every path that is NOT a registered symlink,
//! so most of these mirror `file_path_stat_tests`; the symlink-specific
//! behavior lives in `file_path_symlink_tests`. The unsupported-arch case
//! differs slightly (lstat dropped on ARM64 — newfstatat is the only
//! stat-shaped syscall there).

use super::*;
use super::stat_layouts::{S_IFREG_0755, ST_BLKSIZE};
use super::file_path_tests_support::*;
use crate::memory::Permission;
use crate::state::{FdFlags, RustSimState};
use crate::symbolic::RustBV;
use crate::syscalls::NativeSyscall;

#[test]
fn lstat_unknown_path_returns_minus_one() {
    let mut state = state_with_path(b"/tmp/never-registered");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeLstatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("lstat ok");
    assert_eq!(expect_continue(out), NEG_ONE);
    // Buffer must NOT have been touched on the failure path.
    assert_eq!(read_u64_le(&state, 0x4000), 0);
}

#[test]
fn lstat_empty_path_returns_minus_one() {
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    state
        .memory_store(0x2000, RustBV::concrete(0, 8))
        .expect("nul");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeLstatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("lstat ok");
    assert_eq!(expect_continue(out), NEG_ONE);
}

#[test]
fn lstat_known_path_with_content_writes_amd64_layout() {
    let mut state = state_with_path(b"/tmp/lsized");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let _fd = state
        .file_system()
        .open_with_content("/tmp/lsized".into(), FdFlags::ReadOnly, b"abc".to_vec())
        .expect("fd space is not exhausted in tests");

    let out = NativeLstatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("lstat ok");
    assert_eq!(expect_continue(out), 0);

    // st_size at offset 0x30
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 3);
    // st_mode at offset 0x18 (u32)
    assert_eq!(read_u32_le(&state, 0x4000 + 0x18), S_IFREG_0755 as u32);
    // st_blksize at offset 0x38
    assert_eq!(read_u64_le(&state, 0x4000 + 0x38), ST_BLKSIZE);
}

#[test]
fn lstat_unsupported_arch_falls_back() {
    // ARM64 asm-generic ABI dropped legacy lstat (only newfstatat).
    // MIPS64 carries the legacy 64-bit struct stat with no native writer
    // — handler errors out so the dispatcher falls back to Python's error
    // path. X86 (angr-11djq.5.1), ARM (angr-11djq.5.2) and MIPS32
    // (angr-11djq.5.3, via the LFS lstat64 number) ARE supported, so they
    // are excluded.
    for arch in ["aarch64", "mips64"] {
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();
        state
            .file_system()
            .register_known_path("/tmp/foo".to_string());

        let err = NativeLstatSyscall
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
fn lstat_symbolic_pathname_addr_falls_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    let sym_ptr = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "pathname_ptr", 64)
    };
    let err = NativeLstatSyscall
        .call(&mut state, &[sym_ptr, RustBV::concrete(0x4000, 64)])
        .expect_err("must fall back");
    assert_symbolic_arg(err, "pathname");
}

#[test]
fn lstat_unmapped_buf_surfaces_error() {
    let mut state = state_with_path(b"/tmp/known-l");
    state
        .file_system()
        .register_known_path("/tmp/known-l".to_string());
    // Do NOT map the statbuf page.
    let err = NativeLstatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x8000, 64)],
        )
        .expect_err("unmapped should error");
    assert_memory_err(err);
}

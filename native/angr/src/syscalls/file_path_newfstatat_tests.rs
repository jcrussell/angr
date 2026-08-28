//! `newfstatat` tests (angr-poao). `AT_SYMLINK_NOFOLLOW` handling lives in
//! `file_path_symlink_tests`.

use super::*;
use super::stat_layouts::{S_IFREG_0755, ST_BLKSIZE};
use super::file_path_tests_support::*;
use crate::memory::Permission;
use crate::state::{FdFlags, RustSimState};
use crate::symbolic::RustBV;
use crate::syscalls::NativeSyscall;

#[test]
fn newfstatat_unknown_path_returns_minus_one() {
    let mut state = state_with_path(b"/no/such/file");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeNewfstatatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x4000, 64),
                RustBV::concrete(0, 64), // flag
            ],
        )
        .expect("newfstatat ok");
    assert_eq!(expect_continue(out), NEG_ONE);
    // Buffer must not have been touched on the failure path.
    assert_eq!(read_u64_le(&state, 0x4000), 0);
}

#[test]
fn newfstatat_at_fdcwd_known_path_amd64_layout() {
    let mut state = state_with_path(b"/tmp/nfa.txt");
    state.map_memory(0x4000, 0x1000, Permission::RW);
    let _fd = state
        .file_system()
        .open_with_content(
            "/tmp/nfa.txt".into(),
            FdFlags::ReadOnly,
            b"hello, world!".to_vec(),
        )
        .expect("fd space is not exhausted in tests");

    let out = NativeNewfstatatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x4000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("newfstatat ok");
    assert_eq!(expect_continue(out), 0);

    // AMD64-specific offsets (same as fstat/stat).
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 13);
    assert_eq!(read_u32_le(&state, 0x4000 + 0x18), S_IFREG_0755 as u32);
    assert_eq!(read_u64_le(&state, 0x4000 + 0x38), ST_BLKSIZE);
}

#[test]
fn newfstatat_at_fdcwd_known_path_aarch64_layout() {
    // ARM64 has no legacy lstat/stat — newfstatat (79) is the only
    // stat-shaped syscall on the asm-generic ABI. Critical that the
    // ARM64 struct stat layout is used here.
    let mut state = RustSimState::new("aarch64").expect("state");
    stage_path(&mut state, 0x2000, b"/tmp/arm-nfa");
    state.map_memory(0x4000, 0x1000, Permission::RW);
    let _fd = state
        .file_system()
        .open_with_content("/tmp/arm-nfa".into(), FdFlags::ReadOnly, vec![0u8; 4096])
        .expect("fd space is not exhausted in tests");

    let out = NativeNewfstatatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x4000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("newfstatat ok");
    assert_eq!(expect_continue(out), 0);

    // ARM64-specific: st_mode at 0x10 (u32), st_nlink at 0x14, blksize is u32.
    assert_eq!(read_u32_le(&state, 0x4000 + 0x10), S_IFREG_0755 as u32);
    assert_eq!(read_u32_le(&state, 0x4000 + 0x14), 0);
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 4096);
    assert_eq!(read_u32_le(&state, 0x4000 + 0x38), ST_BLKSIZE as u32);
}

#[test]
fn newfstatat_absolute_path_ignores_dirfd() {
    // Absolute paths bypass dirfd entirely.
    let mut state = RustSimState::new("amd64").expect("state");
    state
        .file_system()
        .register_known_path("/etc/passwd".to_string());
    stage_path(&mut state, 0x2000, b"/etc/passwd");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeNewfstatatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(42, 64), // arbitrary dirfd — absolute path
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x4000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("newfstatat ok");
    assert_eq!(expect_continue(out), 0);
    // Empty content (registered without fd) — st_size = 0.
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 0);
}

#[test]
fn newfstatat_relative_path_non_atfdcwd_returns_minus_one() {
    // Mirrors NativeOpenatSyscall / NativeFaccessatSyscall: we do
    // not model dirfd directories, so relative paths with a
    // non-AT_FDCWD dirfd cannot be resolved.
    let mut state = RustSimState::new("amd64").expect("state");
    state
        .file_system()
        .register_known_path("local.txt".to_string());
    stage_path(&mut state, 0x2000, b"local.txt");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeNewfstatatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(7, 64), // arbitrary dirfd != AT_FDCWD
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x4000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("newfstatat ok");
    assert_eq!(expect_continue(out), NEG_ONE);
    // Buffer must NOT have been touched on the failure path.
    assert_eq!(read_u64_le(&state, 0x4000), 0);
}

#[test]
fn newfstatat_empty_path_returns_minus_one() {
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    state
        .memory_store(0x2000, RustBV::concrete(0, 8))
        .expect("nul");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeNewfstatatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x4000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("newfstatat ok");
    assert_eq!(expect_continue(out), NEG_ONE);
}

#[test]
fn newfstatat_unsupported_arch_falls_back() {
    // MIPS64 carries the legacy 64-bit struct stat with no native writer.
    // Handler errors out before any memory read. X86 (angr-11djq.5.1),
    // ARM (angr-11djq.5.2) and MIPS32 (angr-11djq.5.3, via fstatat64 4293)
    // ARE supported, so they are excluded.
    {
        let arch = "mips64";
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();
        state
            .file_system()
            .register_known_path("/tmp/foo".to_string());

        let err = NativeNewfstatatSyscall
            .call(
                &mut state,
                &[
                    // bits == 64 here (mips64); the value already fits in 64
                    // bits, so no masking is needed. `(1u64 << 64) - 1` would
                    // panic on shift-overflow in a debug build (angr-yojqz),
                    // matching the unmasked usage in the sibling arch tests.
                    RustBV::concrete(TEST_AT_FDCWD as u128, bits),
                    RustBV::concrete(0x2000, bits),
                    RustBV::concrete(0x4000, bits),
                    RustBV::concrete(0, bits),
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
fn newfstatat_symbolic_dirfd_falls_back() {
    let mut state = state_with_path(b"/tmp/x");
    let sym_dirfd = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "dirfd", 64)
    };

    let err = NativeNewfstatatSyscall
        .call(
            &mut state,
            &[
                sym_dirfd,
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x4000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect_err("must fall back");
    assert_symbolic_arg(err, "dirfd");
}

#[test]
fn newfstatat_symbolic_pathname_addr_falls_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    let sym_ptr = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "pathname_ptr", 64)
    };
    let err = NativeNewfstatatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                sym_ptr,
                RustBV::concrete(0x4000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect_err("must fall back");
    assert_symbolic_arg(err, "pathname");
}

#[test]
fn newfstatat_unmapped_buf_surfaces_error() {
    let mut state = state_with_path(b"/tmp/known-nfa");
    state
        .file_system()
        .register_known_path("/tmp/known-nfa".to_string());
    // Do NOT map the statbuf page.
    let err = NativeNewfstatatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x8000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect_err("unmapped should error");
    assert_memory_err(err);
}

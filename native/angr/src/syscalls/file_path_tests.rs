//! `open` / `openat` / `close` / `access` / `faccessat` path-resolution tests.
//!
//! angr-7hwz pattern: extracted out of `file_path.rs`'s former in-file
//! `mod tests`; split further by handler family in angr-5mnx3.70. Declared as
//! a direct child of `file_path` so `use super::*` reaches its private items.

use super::*;
use super::file_path_tests_support::*;
use crate::memory::Permission;
use crate::state::{FdFlags, RustSimState};
use crate::symbolic::{RustBV, SymContext};
use crate::syscalls::{NativeSyscall, SyscallOutcome};


// angr-0hif.1 stub-sweep deleted — every file_path stub has now
// been promoted: faccessat (angr-6009), lstat/newfstatat (angr-poao),
// readlink/readlinkat (angr-wv38). Per-handler semantics are pinned
// by the dedicated tests below.

#[test]
fn open_allocates_fresh_fd_and_records_name() {
    let mut state = state_with_path(b"/tmp/example.txt");

    let outcome = NativeOpenSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64), // O_RDONLY
                RustBV::concrete(0, 64),
            ],
        )
        .expect("open ok");
    let fd = expect_continue(outcome);
    assert_eq!(fd, 3, "first allocated fd should be 3");
    assert!(state.file_system_ref().is_open(fd as u32));
    let (name, _, _, _, _) = state.file_system_ref().fd_info(fd as u32).unwrap();
    assert_eq!(name, "/tmp/example.txt");
}

#[test]
fn open_empty_path_returns_minus_one() {
    let mut state = RustSimState::new("amd64").expect("state");
    // Just stage a NUL at addr 0x2000.
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    state
        .memory_store(0x2000, RustBV::concrete(0, 8))
        .expect("store nul");

    let out = NativeOpenSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();
    assert_eq!(expect_continue(out), NEG_ONE);
}

#[test]
fn open_symbolic_path_byte_falls_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let sym_byte = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "first_path_byte", 8)
    };
    state.memory_store(0x2000, sym_byte).unwrap();

    let err = NativeOpenSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect_err("must fall back");
    assert_symbolic_arg(err, "open");
}

#[test]
fn open_symbolic_pathname_addr_falls_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    let sym_ptr = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "pathname_ptr", 64)
    };
    let err = NativeOpenSyscall
        .call(
            &mut state,
            &[sym_ptr, RustBV::concrete(0, 64), RustBV::concrete(0, 64)],
        )
        .expect_err("must fall back");
    assert_symbolic_arg(err, "pathname");
}

#[test]
fn openat_absolute_path_allocates_fd_ignoring_dirfd() {
    let mut state = state_with_path(b"/etc/hosts");

    let outcome = NativeOpenatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(99, 64), // arbitrary dirfd — ignored for absolute paths
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("openat ok");
    match outcome {
        SyscallOutcome::Continue { ret } => {
            assert_eq!(ret, 3);
            assert!(state.file_system_ref().is_open(ret as u32));
        }
        other => panic!("expected Continue, got {other:?}"),
    }
}

#[test]
fn openat_relative_path_with_at_fdcwd_allocates_fd() {
    let mut state = state_with_path(b"flag.txt");

    let outcome = NativeOpenatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(AT_FDCWD_UNSIGNED as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("openat ok");
    assert_eq!(expect_continue(outcome), 3);
}

#[test]
fn openat_relative_path_without_at_fdcwd_returns_minus_one() {
    let mut state = state_with_path(b"flag.txt");
    let prev_next_fd = state.file_system_ref().next_fd();

    let outcome = NativeOpenatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(7, 64), // arbitrary dirfd ≠ AT_FDCWD
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("openat ok");
    assert_eq!(expect_continue(outcome), NEG_ONE);
    // No fd should have been allocated.
    assert_eq!(state.file_system_ref().next_fd(), prev_next_fd);
}

#[test]
fn close_open_fd_returns_zero_and_marks_closed() {
    let mut state = RustSimState::new("amd64").expect("state");
    // Allocate a fresh fd via the file system directly.
    let fd = state
        .file_system()
        .open("/tmp/x".into(), FdFlags::ReadOnly)
        .expect("fd space is not exhausted in tests");
    assert!(state.file_system_ref().is_open(fd));

    let outcome = NativeCloseSyscall
        .call(&mut state, &[RustBV::concrete(fd as u128, 64)])
        .unwrap();
    assert_eq!(expect_continue(outcome), 0);
    assert!(!state.file_system_ref().is_open(fd));
}

#[test]
fn close_unknown_fd_returns_minus_one() {
    let mut state = RustSimState::new("amd64").expect("state");
    // fd=99 was never allocated.
    let outcome = NativeCloseSyscall
        .call(&mut state, &[RustBV::concrete(99, 64)])
        .unwrap();
    assert_eq!(expect_continue(outcome), NEG_ONE);
}

#[test]
fn close_symbolic_fd_falls_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    let ctx = SymContext::new();
    let sym_fd = RustBV::symbolic(&ctx, "fd", 64);
    let err = NativeCloseSyscall
        .call(&mut state, &[sym_fd])
        .expect_err("must fall back");
    assert_symbolic_arg(err, "fd");
}

#[test]
fn access_unknown_path_returns_minus_one() {
    let mut state = state_with_path(b"/no/such/file");

    let out = NativeAccessSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0, 64)],
        )
        .expect("access ok");
    assert_eq!(expect_continue(out), NEG_ONE);
}

#[test]
fn access_after_open_returns_zero() {
    let mut state = state_with_path(b"/tmp/exists.txt");

    // Open registers the path.
    NativeOpenSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("open ok");

    // Re-stage path at a different addr to prove access reads it
    // fresh (not relying on caller-side cached state).
    stage_path(&mut state, 0x3000, b"/tmp/exists.txt");

    let out = NativeAccessSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x3000, 64), RustBV::concrete(0, 64)],
        )
        .expect("access ok");
    assert_eq!(expect_continue(out), 0);
}

#[test]
fn access_registered_path_returns_zero() {
    let mut state = RustSimState::new("amd64").expect("state");
    // Mirror the Python state.fs.insert path: register without
    // allocating an fd.
    state
        .file_system()
        .register_known_path("/etc/passwd".to_string());
    stage_path(&mut state, 0x2000, b"/etc/passwd");

    let out = NativeAccessSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0, 64)],
        )
        .expect("access ok");
    assert_eq!(expect_continue(out), 0);
}

#[test]
fn access_empty_path_returns_minus_one() {
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    state
        .memory_store(0x2000, RustBV::concrete(0, 8))
        .expect("store NUL");

    let out = NativeAccessSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0, 64)],
        )
        .expect("access ok");
    assert_eq!(expect_continue(out), NEG_ONE);
}

#[test]
fn access_symbolic_path_byte_falls_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let sym_byte = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "first_path_byte", 8)
    };
    state.memory_store(0x2000, sym_byte).unwrap();

    let err = NativeAccessSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0, 64)],
        )
        .expect_err("must fall back");
    assert_symbolic_arg(err, "access");
}

#[test]
fn access_symbolic_pathname_addr_falls_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    let sym_ptr = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "pathname_ptr", 64)
    };
    let err = NativeAccessSyscall
        .call(&mut state, &[sym_ptr, RustBV::concrete(0, 64)])
        .expect_err("must fall back");
    assert_symbolic_arg(err, "pathname");
}

#[test]
fn access_round_trip_sweeps_supported_arches() {
    // AArch64 has no legacy `access` syscall (asm-generic only ships
    // faccessat), but the handler itself is arch-agnostic — exercise
    // it from each `RustSimState::new(...)` arch to confirm the
    // path-read + lookup path is independent of pointer width.
    for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();
        stage_path(&mut state, 0x2000, b"/tmp/access-rt");

        // Unknown → -1.
        let out = NativeAccessSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, bits), RustBV::concrete(0, bits)],
            )
            .expect("access");
        match out {
            SyscallOutcome::Continue { ret } => {
                assert_eq!(ret, NEG_ONE, "{arch} pre-open")
            }
            other => panic!("{arch} pre-open: got {other:?}"),
        }

        // Register, then known → 0.
        state
            .file_system()
            .register_known_path("/tmp/access-rt".to_string());
        let out2 = NativeAccessSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, bits), RustBV::concrete(0, bits)],
            )
            .expect("access");
        match out2 {
            SyscallOutcome::Continue { ret } => {
                assert_eq!(ret, 0, "{arch} post-register")
            }
            other => panic!("{arch} post-register: got {other:?}"),
        }
    }
}
#[test]
fn faccessat_unknown_path_returns_minus_one() {
    let mut state = state_with_path(b"/no/such/file");

    let out = NativeFaccessatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64), // mode (ignored)
            ],
        )
        .expect("faccessat ok");
    assert_eq!(expect_continue(out), NEG_ONE);
}

#[test]
fn faccessat_after_open_returns_zero_with_at_fdcwd() {
    let mut state = state_with_path(b"/tmp/fa-exists.txt");

    NativeOpenSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("open ok");

    let out = NativeFaccessatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("faccessat ok");
    assert_eq!(expect_continue(out), 0);
}

#[test]
fn faccessat_absolute_path_ignores_dirfd() {
    // Absolute paths bypass dirfd entirely — any value should resolve.
    let mut state = RustSimState::new("amd64").expect("state");
    state
        .file_system()
        .register_known_path("/etc/passwd".to_string());
    stage_path(&mut state, 0x2000, b"/etc/passwd");

    let out = NativeFaccessatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(42, 64), // arbitrary dirfd
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("faccessat ok");
    assert_eq!(expect_continue(out), 0);
}

#[test]
fn faccessat_relative_path_non_atfdcwd_returns_minus_one() {
    // Mirrors NativeOpenatSyscall: we do not model dirfd directories,
    // so relative paths with a non-AT_FDCWD dirfd cannot be resolved.
    let mut state = RustSimState::new("amd64").expect("state");
    // Register the bare name in case the handler ever resolved it
    // without consulting dirfd — proves we are NOT doing that.
    state
        .file_system()
        .register_known_path("local.txt".to_string());
    stage_path(&mut state, 0x2000, b"local.txt");

    let out = NativeFaccessatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(7, 64), // arbitrary dirfd != AT_FDCWD
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("faccessat ok");
    assert_eq!(expect_continue(out), NEG_ONE);
}

#[test]
fn faccessat_empty_path_returns_minus_one() {
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    state
        .memory_store(0x2000, RustBV::concrete(0, 8))
        .expect("store NUL");

    let out = NativeFaccessatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("faccessat ok");
    assert_eq!(expect_continue(out), NEG_ONE);
}

#[test]
fn faccessat_symbolic_path_byte_falls_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let sym_byte = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "first_path_byte", 8)
    };
    state.memory_store(0x2000, sym_byte).unwrap();

    let err = NativeFaccessatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect_err("must fall back");
    assert_symbolic_arg(err, "faccessat");
}

#[test]
fn faccessat_symbolic_dirfd_falls_back() {
    let mut state = state_with_path(b"/tmp/x");
    let sym_dirfd = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "dirfd", 64)
    };

    let err = NativeFaccessatSyscall
        .call(
            &mut state,
            &[
                sym_dirfd,
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect_err("must fall back");
    assert_symbolic_arg(err, "dirfd");
}

#[test]
fn faccessat_round_trip_sweeps_supported_arches() {
    // Like access_round_trip_sweeps_supported_arches but for the
    // *at variant — arch independence of the dispatch + lookup paths.
    for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();
        stage_path(&mut state, 0x2000, b"/tmp/faccessat-rt");
        // AT_FDCWD truncates to the arch's pointer width — the raw
        // constant is 32-bit signed -100 reinterpreted unsigned, which
        // is the same value Python's openat.py uses on every arch.
        let atfdcwd = if bits == 64 {
            TEST_AT_FDCWD as u128
        } else {
            (TEST_AT_FDCWD & ((1u64 << bits) - 1)) as u128
        };

        let out = NativeFaccessatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(atfdcwd, bits),
                    RustBV::concrete(0x2000, bits),
                    RustBV::concrete(0, bits),
                ],
            )
            .expect("faccessat");
        match out {
            SyscallOutcome::Continue { ret } => {
                assert_eq!(ret, NEG_ONE, "{arch} pre-register")
            }
            other => panic!("{arch} pre-register: got {other:?}"),
        }

        state
            .file_system()
            .register_known_path("/tmp/faccessat-rt".to_string());
        let out2 = NativeFaccessatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(atfdcwd, bits),
                    RustBV::concrete(0x2000, bits),
                    RustBV::concrete(0, bits),
                ],
            )
            .expect("faccessat");
        match out2 {
            SyscallOutcome::Continue { ret } => {
                assert_eq!(ret, 0, "{arch} post-register")
            }
            other => panic!("{arch} post-register: got {other:?}"),
        }
    }
}

#[test]
fn open_close_round_trip_sweeps_supported_arches() {
    for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();
        stage_path(&mut state, 0x2000, b"/tmp/roundtrip");

        let fd_out = NativeOpenSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, bits),
                    RustBV::concrete(0, bits),
                    RustBV::concrete(0, bits),
                ],
            )
            .expect("open");
        let fd = match fd_out {
            SyscallOutcome::Continue { ret } => ret,
            other => panic!("{arch} open: expected Continue, got {other:?}"),
        };
        assert!(state.file_system_ref().is_open(fd as u32), "{arch}");

        let close_out = NativeCloseSyscall
            .call(&mut state, &[RustBV::concrete(fd as u128, bits)])
            .expect("close");
        match close_out {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 0, "{arch}"),
            other => panic!("{arch} close: expected Continue, got {other:?}"),
        }
        assert!(!state.file_system_ref().is_open(fd as u32), "{arch}");
    }
}

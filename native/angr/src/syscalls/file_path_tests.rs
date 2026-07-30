// angr-7hwz pattern: syscall path-resolution unit tests, extracted out of
// the former in-file `mod tests` (~1733 lines) into a sibling file to shrink
// syscalls/file_path.rs below the god-object threshold. Declared as a direct
// child of `file_path` so `use super::*` reaches its private items.

use super::*;
use crate::memory::Permission;
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};
use crate::syscalls::{NativeSyscall, SyscallOutcome};

/// Map an RW page at 0x2000 and stage a NUL-terminated path byte-by-byte.
fn stage_path(state: &mut RustSimState, addr: u64, path: &[u8]) {
    state.map_memory(addr & !0xfff, 0x1000, Permission::RWX);
    for (i, b) in path.iter().enumerate() {
        state
            .memory_store(addr + i as u64, RustBV::concrete(*b as u128, 8))
            .expect("store path byte");
    }
    state
        .memory_store(addr + path.len() as u64, RustBV::concrete(0, 8))
        .expect("store NUL");
}

/// Build an amd64 state with `path` staged (NUL-terminated) at 0x2000 —
/// the default staging address shared by most file_path syscall tests.
fn state_with_path(path: &[u8]) -> RustSimState {
    let mut state = RustSimState::new("amd64").expect("state");
    stage_path(&mut state, 0x2000, path);
    state
}

/// Unwrap a `Continue` outcome's return value, panicking with context
/// otherwise.
fn expect_continue(outcome: SyscallOutcome) -> u64 {
    match outcome {
        SyscallOutcome::Continue { ret } => ret,
        other => panic!("expected Continue, got {other:?}"),
    }
}

/// Assert that `err` is a `SymbolicArgument` whose message contains
/// `needle`.
fn assert_symbolic_arg(err: SyscallError, needle: &str) {
    match err {
        SyscallError::SymbolicArgument(msg) => assert!(
            msg.contains(needle),
            "expected SymbolicArgument message to contain {needle:?}, got {msg:?}",
        ),
        other => panic!("expected SymbolicArgument, got {other:?}"),
    }
}

/// Assert that `err` is a `Memory` error (message not inspected).
fn assert_memory_err(err: SyscallError) {
    match err {
        SyscallError::Memory(_) => {}
        other => panic!("expected Memory error, got {other:?}"),
    }
}

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
    let fd = state.file_system().open("/tmp/x".into(), FdFlags::ReadOnly);
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

/// `AT_FDCWD` reinterpreted as unsigned 64-bit — same constant the
/// handler matches on. Kept inline so the test stays self-contained.
const TEST_AT_FDCWD: u64 = AT_FDCWD_UNSIGNED;

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

// ---- readlink / readlinkat (angr-wv38) ----

#[test]
fn readlink_unknown_path_returns_minus_one() {
    let mut state = state_with_path(b"/no/such/path");

    let out = NativeReadlinkSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x3000, 64), // buf — must NOT be touched
                RustBV::concrete(256, 64),    // bufsiz
            ],
        )
        .expect("readlink ok");
    assert_eq!(expect_continue(out), NEG_ONE);
}

#[test]
fn readlink_known_path_still_returns_minus_one() {
    // Even for paths the FileSystem knows about, readlink must return
    // -1 (EINVAL — not a symlink). The FileSystem has no symlinks.
    let mut state = state_with_path(b"/tmp/known");
    state
        .file_system()
        .register_known_path("/tmp/known".to_string());

    let out = NativeReadlinkSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x3000, 64),
                RustBV::concrete(256, 64),
            ],
        )
        .expect("readlink ok");
    assert_eq!(expect_continue(out), NEG_ONE);
}

#[test]
fn readlink_empty_path_returns_minus_one() {
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    state
        .memory_store(0x2000, RustBV::concrete(0, 8))
        .expect("store nul");

    let out = NativeReadlinkSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("readlink ok");
    assert_eq!(expect_continue(out), NEG_ONE);
}

#[test]
fn readlink_empty_path_guard_beats_empty_key_symlink() {
    // angr-myzjx.13: the empty-path guard must return -1 even when a
    // symlink is registered under the empty-string key — matching
    // readlinkat's guard. Without the guard, readlink("") would resolve
    // the target and diverge from readlinkat(AT_FDCWD, "", ...).
    let target = b"/should/not/resolve";
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    state
        .memory_store(0x2000, RustBV::concrete(0, 8))
        .expect("store nul");
    state
        .file_system()
        .add_symlink(String::new(), target.to_vec());
    state.map_memory(0x3000, 0x1000, Permission::RWX);

    let out = NativeReadlinkSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x3000, 64),
                RustBV::concrete(256, 64),
            ],
        )
        .expect("readlink ok");
    assert_eq!(expect_continue(out), NEG_ONE);
}

#[test]
fn readlink_buf_is_not_modified_on_failure() {
    // The buffer must NOT be written: real Linux only fills it on a
    // positive return, and we always return -1.
    let mut state = state_with_path(b"/whatever");
    // Pre-mark buf with a sentinel; after the call it must still be
    // there (we never wrote to it).
    state.map_memory(0x3000, 0x1000, Permission::RWX);
    for i in 0..8 {
        state
            .memory_store(0x3000 + i, RustBV::concrete(0xAA, 8))
            .expect("store sentinel");
    }

    let _ = NativeReadlinkSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x3000, 64),
                RustBV::concrete(8, 64),
            ],
        )
        .expect("readlink ok");

    for i in 0..8u64 {
        let bv = state.memory_load(0x3000 + i, 1).expect("load");
        assert_eq!(
            bv.as_u64().unwrap(),
            0xAA,
            "buf byte {i} was touched (must be untouched on -1 return)"
        );
    }
}

#[test]
fn readlink_registered_symlink_writes_target() {
    // angr-11djq.6.2: a path registered in the symlink table returns its
    // target bytes (no NUL terminator) and `target_len` as the count.
    let target = b"/real/destination";
    let mut state = state_with_path(b"/link");
    state
        .file_system()
        .add_symlink("/link".to_string(), target.to_vec());
    state.map_memory(0x3000, 0x1000, Permission::RWX);

    let out = NativeReadlinkSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x3000, 64), // buf
                RustBV::concrete(256, 64),    // bufsiz (>> target len)
            ],
        )
        .expect("readlink ok");
    assert_eq!(expect_continue(out), target.len() as u64);
    for (i, b) in target.iter().enumerate() {
        let got = state.memory_load(0x3000 + i as u64, 1).expect("load");
        assert_eq!(got.as_u64().unwrap() as u8, *b, "target byte {i}");
    }
    // No NUL terminator: the byte just past the target must be untouched
    // (memory_load of an unwritten cell would be symbolic/zero, but the
    // contract is we wrote exactly target.len() bytes — assert the count).
}

#[test]
fn readlink_symlink_truncates_to_bufsiz() {
    // bufsiz smaller than the target → write only `bufsiz` bytes and
    // return `bufsiz` (matches readlink(2) truncation, no error).
    let target = b"/very/long/symlink/target";
    let mut state = state_with_path(b"/link");
    state
        .file_system()
        .add_symlink("/link".to_string(), target.to_vec());
    state.map_memory(0x3000, 0x1000, Permission::RWX);
    // Pre-fill buf with a sentinel so we can confirm only `bufsiz` bytes
    // were overwritten.
    for i in 0..target.len() as u64 {
        state
            .memory_store(0x3000 + i, RustBV::concrete(0xAA, 8))
            .expect("sentinel");
    }

    let bufsiz = 5u64;
    let out = NativeReadlinkSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x3000, 64),
                RustBV::concrete(bufsiz as u128, 64),
            ],
        )
        .expect("readlink ok");
    assert_eq!(expect_continue(out), bufsiz);
    for i in 0..bufsiz {
        let got = state.memory_load(0x3000 + i, 1).expect("load");
        assert_eq!(got.as_u64().unwrap() as u8, target[i as usize]);
    }
    // Byte at index `bufsiz` must still be the sentinel (not written).
    let untouched = state.memory_load(0x3000 + bufsiz, 1).expect("load");
    assert_eq!(untouched.as_u64().unwrap(), 0xAA, "wrote past bufsiz");
}

#[test]
fn readlinkat_registered_symlink_writes_target() {
    // angr-11djq.6.2: readlinkat via AT_FDCWD on a registered symlink
    // writes the target and returns its length, like readlink.
    let target = b"/at/destination";
    let mut state = state_with_path(b"/atlink");
    state
        .file_system()
        .add_symlink("/atlink".to_string(), target.to_vec());
    state.map_memory(0x3000, 0x1000, Permission::RWX);

    let out = NativeReadlinkatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(AT_FDCWD_UNSIGNED as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x3000, 64),
                RustBV::concrete(256, 64),
            ],
        )
        .expect("readlinkat ok");
    assert_eq!(expect_continue(out), target.len() as u64);
    for (i, b) in target.iter().enumerate() {
        let got = state.memory_load(0x3000 + i as u64, 1).expect("load");
        assert_eq!(got.as_u64().unwrap() as u8, *b, "target byte {i}");
    }
}

#[test]
fn readlink_symbolic_path_byte_falls_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let sym_byte = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "first_path_byte", 8)
    };
    state.memory_store(0x2000, sym_byte).unwrap();

    let err = NativeReadlinkSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect_err("must fall back");
    assert_symbolic_arg(err, "readlink");
}

#[test]
fn readlink_symbolic_pathname_addr_falls_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    let sym_ptr = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "pathname_ptr", 64)
    };
    let err = NativeReadlinkSyscall
        .call(
            &mut state,
            &[sym_ptr, RustBV::concrete(0, 64), RustBV::concrete(0, 64)],
        )
        .expect_err("must fall back");
    assert_symbolic_arg(err, "pathname");
}

#[test]
fn readlink_round_trip_sweeps_supported_arches() {
    // readlink is registered on every arch except ARM64; the handler
    // itself has no arch-specific code. Sweep all arches it can be
    // dispatched on to pin arch-independence.
    for arch in ["amd64", "x86", "armel", "mipsel"] {
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();
        stage_path(&mut state, 0x2000, b"/tmp/whatever");
        let out = NativeReadlinkSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, bits),
                    RustBV::concrete(0, bits),
                    RustBV::concrete(0, bits),
                ],
            )
            .expect("readlink");
        match out {
            SyscallOutcome::Continue { ret } => {
                assert_eq!(ret, NEG_ONE, "{arch} readlink ret")
            }
            other => panic!("{arch}: got {other:?}"),
        }
    }
}

#[test]
fn readlinkat_at_fdcwd_unknown_path_returns_minus_one() {
    let mut state = state_with_path(b"/no/such/path");

    let out = NativeReadlinkatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x3000, 64),
                RustBV::concrete(256, 64),
            ],
        )
        .expect("readlinkat ok");
    assert_eq!(expect_continue(out), NEG_ONE);
}

#[test]
fn readlinkat_known_path_still_returns_minus_one() {
    let mut state = state_with_path(b"/tmp/known");
    state
        .file_system()
        .register_known_path("/tmp/known".to_string());

    let out = NativeReadlinkatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x3000, 64),
                RustBV::concrete(256, 64),
            ],
        )
        .expect("readlinkat ok");
    assert_eq!(expect_continue(out), NEG_ONE);
}

#[test]
fn readlinkat_absolute_path_ignores_dirfd() {
    // Absolute path: dirfd does not matter, still -1.
    let mut state = state_with_path(b"/abs/path");

    let out = NativeReadlinkatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(99, 64), // arbitrary dirfd
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x3000, 64),
                RustBV::concrete(256, 64),
            ],
        )
        .expect("readlinkat ok");
    assert_eq!(expect_continue(out), NEG_ONE);
}

#[test]
fn readlinkat_relative_path_non_atfdcwd_returns_minus_one() {
    // Relative path with arbitrary dirfd: still -1 (would be -1
    // anyway, but the short-circuit branch exists for symmetry with
    // faccessat / openat).
    let mut state = state_with_path(b"relative/path");

    let out = NativeReadlinkatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(99, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x3000, 64),
                RustBV::concrete(256, 64),
            ],
        )
        .expect("readlinkat ok");
    assert_eq!(expect_continue(out), NEG_ONE);
}

#[test]
fn readlinkat_empty_path_returns_minus_one() {
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    state
        .memory_store(0x2000, RustBV::concrete(0, 8))
        .expect("store nul");

    let out = NativeReadlinkatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("readlinkat ok");
    assert_eq!(expect_continue(out), NEG_ONE);
}

#[test]
fn readlinkat_symbolic_dirfd_falls_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    let sym_dirfd = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "dirfd", 64)
    };
    let err = NativeReadlinkatSyscall
        .call(
            &mut state,
            &[
                sym_dirfd,
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect_err("must fall back");
    assert_symbolic_arg(err, "dirfd");
}

#[test]
fn readlinkat_symbolic_pathname_addr_falls_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    let sym_ptr = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "pathname_ptr", 64)
    };
    let err = NativeReadlinkatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                sym_ptr,
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect_err("must fall back");
    assert_symbolic_arg(err, "pathname");
}

#[test]
fn readlinkat_round_trip_sweeps_supported_arches() {
    // readlinkat is registered on every arch including ARM64 (78).
    for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();
        stage_path(&mut state, 0x2000, b"/tmp/whatever");
        let atfdcwd = if bits == 64 {
            TEST_AT_FDCWD as u128
        } else {
            (TEST_AT_FDCWD & ((1u64 << bits) - 1)) as u128
        };
        let out = NativeReadlinkatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(atfdcwd, bits),
                    RustBV::concrete(0x2000, bits),
                    RustBV::concrete(0, bits),
                    RustBV::concrete(0, bits),
                ],
            )
            .expect("readlinkat");
        match out {
            SyscallOutcome::Continue { ret } => {
                assert_eq!(ret, NEG_ONE, "{arch} readlinkat ret")
            }
            other => panic!("{arch}: got {other:?}"),
        }
    }
}

/// Read `size` bytes of LE-packed u64 from memory at `addr`.
fn read_u64_le(state: &RustSimState, addr: u64) -> u64 {
    let bv = state.memory_load(addr, 8).expect("load");
    bv.as_u64().expect("concrete")
}

fn read_u32_le(state: &RustSimState, addr: u64) -> u32 {
    let bv = state.memory_load(addr, 4).expect("load");
    bv.as_u64().expect("concrete") as u32
}

#[test]
fn fstat_unknown_fd_returns_minus_one() {
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeFstatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(99, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("fstat ok");
    assert_eq!(expect_continue(out), NEG_ONE);
    // Buffer must NOT have been touched on the failure path
    // (read 0s from the freshly-mapped page).
    assert_eq!(read_u64_le(&state, 0x4000), 0);
}

#[test]
fn fstat_known_fd_writes_amd64_layout_and_returns_zero() {
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    // Seed a file with concrete content so content_len = 13.
    let fd = state.file_system().open_with_content(
        "/tmp/hello".into(),
        FdFlags::ReadOnly,
        b"hello, world!".to_vec(),
    );

    let out = NativeFstatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 64),
                RustBV::concrete(0x4000, 64),
            ],
        )
        .expect("fstat ok");
    assert_eq!(expect_continue(out), 0);

    // st_size at offset 0x30
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 13);
    // st_mode at offset 0x18 (u32)
    assert_eq!(read_u32_le(&state, 0x4000 + 0x18), S_IFREG_0755 as u32);
    // st_blksize at offset 0x38
    assert_eq!(read_u64_le(&state, 0x4000 + 0x38), ST_BLKSIZE);
    // st_dev at offset 0 — zero
    assert_eq!(read_u64_le(&state, 0x4000), 0);
    // st_ctimensec at 0x70 — zero
    assert_eq!(read_u64_le(&state, 0x4000 + 0x70), 0);
}

/// angr-0xyq2 Phase 1: an fd whose bounded symbolic content came from the
/// `file_contents` registry reports the symbolic byte count as st_size
/// (the concrete buffer is empty — `effective_size` must not read it).
#[test]
fn fstat_symbolic_content_fd_reports_symbolic_len() {
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let bytes: Vec<RustBV> = (0..21u64)
        .map(|i| RustBV::symbolic_with_id(0x6f000 + i, format!("fstat_sym_{i}"), 8))
        .collect();
    state
        .file_system()
        .register_file_content("/tmp/symflag", bytes);
    let fd = state
        .file_system()
        .open("/tmp/symflag".into(), FdFlags::ReadOnly);

    let out = NativeFstatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 64),
                RustBV::concrete(0x4000, 64),
            ],
        )
        .expect("fstat ok");
    assert_eq!(expect_continue(out), 0);

    // st_size at offset 0x30 — the symbolic byte count.
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 21);
    assert_eq!(read_u32_le(&state, 0x4000 + 0x18), S_IFREG_0755 as u32);
}

#[test]
fn fstat_known_fd_writes_aarch64_layout_and_returns_zero() {
    let mut state = RustSimState::new("aarch64").expect("state");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let fd = state.file_system().open_with_content(
        "/tmp/arm".into(),
        FdFlags::ReadOnly,
        vec![0u8; 4096],
    );

    let out = NativeFstatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 64),
                RustBV::concrete(0x4000, 64),
            ],
        )
        .expect("fstat ok");
    assert_eq!(expect_continue(out), 0);

    // AArch64-specific: st_mode at 0x10 (NOT 0x18 like AMD64).
    assert_eq!(read_u32_le(&state, 0x4000 + 0x10), S_IFREG_0755 as u32);
    // st_nlink at 0x14 — zero u32
    assert_eq!(read_u32_le(&state, 0x4000 + 0x14), 0);
    // st_size at 0x30
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 4096);
    // st_blksize at 0x38 is u32 here
    assert_eq!(read_u32_le(&state, 0x4000 + 0x38), ST_BLKSIZE as u32);
    // The last padding word at 0x78 — zero
    assert_eq!(read_u64_le(&state, 0x4000 + 0x78), 0);
}

#[test]
fn fstat_known_fd_writes_i386_layout_and_returns_zero() {
    // i386 fstat64 / struct stat64 layout (`write_i386_stat`), mirroring
    // angr/procedures/linux_kernel/fstat64.py::_store_i386. Offsets differ
    // from AMD64/AArch64: st_mode at 0x10, st_size at 0x2C, st_blksize at
    // 0x34. Buffers are 32-bit on X86 so the args are sized accordingly.
    let mut state = RustSimState::new("x86").expect("state");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let fd = state.file_system().open_with_content(
        "/tmp/i386".into(),
        FdFlags::ReadOnly,
        b"hello, world!".to_vec(),
    );

    let out = NativeFstatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 32),
                RustBV::concrete(0x4000, 32),
            ],
        )
        .expect("fstat ok");
    assert_eq!(expect_continue(out), 0);

    // st_mode at 0x10 (u32) — concrete S_IFREG | 0o755.
    assert_eq!(read_u32_le(&state, 0x4000 + 0x10), S_IFREG_0755 as u32);
    // st_size at 0x2C (u64).
    assert_eq!(read_u64_le(&state, 0x4000 + 0x2C), 13);
    // st_blksize at 0x34 (low word of the 64-bit store).
    assert_eq!(read_u32_le(&state, 0x4000 + 0x34), ST_BLKSIZE as u32);
    // st_dev at 0 — zero.
    assert_eq!(read_u64_le(&state, 0x4000), 0);
    // st_uid at 0x18 — zero (the overlapping st_nlink 64-bit store is 0).
    assert_eq!(read_u32_le(&state, 0x4000 + 0x18), 0);
}

#[test]
fn fstat_known_fd_writes_arm_layout_and_returns_zero() {
    // ARM (32-bit EABI) fstat64 / struct stat64 layout (`write_arm_stat`),
    // mirroring angr/procedures/linux_kernel/fstat64.py::_store_arm.
    // Offsets differ from i386: st_size at 0x30 (vs 0x2C), st_blksize at
    // 0x38 (vs 0x34), no padding store. Buffers are 32-bit on ARM.
    let mut state = RustSimState::new("armel").expect("state");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let fd = state.file_system().open_with_content(
        "/tmp/arm".into(),
        FdFlags::ReadOnly,
        b"hello, world!".to_vec(),
    );

    let out = NativeFstatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 32),
                RustBV::concrete(0x4000, 32),
            ],
        )
        .expect("fstat ok");
    assert_eq!(expect_continue(out), 0);

    // st_mode at 0x10 (u32) — concrete S_IFREG | 0o755.
    assert_eq!(read_u32_le(&state, 0x4000 + 0x10), S_IFREG_0755 as u32);
    // st_size at 0x30 (u64).
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 13);
    // st_blksize at 0x38 (low word of the 64-bit store).
    assert_eq!(read_u32_le(&state, 0x4000 + 0x38), ST_BLKSIZE as u32);
    // st_dev at 0 — zero.
    assert_eq!(read_u64_le(&state, 0x4000), 0);
    // st_uid at 0x18 — zero (the overlapping st_nlink 64-bit store is 0).
    assert_eq!(read_u32_le(&state, 0x4000 + 0x18), 0);
    // st_ino verification copy at 0x60 — zero.
    assert_eq!(read_u64_le(&state, 0x4000 + 0x60), 0);
}

#[test]
fn fstat_known_fd_writes_mips32_layout_and_returns_zero() {
    // MIPS32 O32 fstat64 / struct stat64 layout (`write_mips32_stat`),
    // mirroring angr/procedures/linux_kernel/fstat64.py::_store_mips32.
    // Notable differences from i386/ARM: st_size at 0x30, st_blksize at
    // 0x50, and NO st_mode field is written (angr's MIPS layout omits it).
    // "mipsel" gives a little-endian MIPS32 so the read_u*_le helpers apply.
    let mut state = RustSimState::new("mipsel").expect("state");
    assert_eq!(state.arch().name(), "MIPS32");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let fd = state.file_system().open_with_content(
        "/tmp/mips".into(),
        FdFlags::ReadOnly,
        b"hello, world!".to_vec(),
    );

    let out = NativeFstatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 32),
                RustBV::concrete(0x4000, 32),
            ],
        )
        .expect("fstat ok");
    assert_eq!(expect_continue(out), 0);

    // st_size at 0x30 (u64).
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 13);
    // st_blksize at 0x50 (low word of the 64-bit store).
    assert_eq!(read_u32_le(&state, 0x4000 + 0x50), ST_BLKSIZE as u32);
    // st_dev at 0 — zero.
    assert_eq!(read_u64_le(&state, 0x4000), 0);
    // st_ino at 0x10 — zero.
    assert_eq!(read_u64_le(&state, 0x4000 + 0x10), 0);
    // st_blocks at 0x58 — zero.
    assert_eq!(read_u64_le(&state, 0x4000 + 0x58), 0);
    // MIPS layout omits st_mode entirely — the byte where i386/ARM put it
    // (0x10) is covered by the st_ino store and stays zero (NOT S_IFREG).
    assert_eq!(read_u32_le(&state, 0x4000 + 0x10), 0);
}

#[test]
fn fstat_unsupported_arch_falls_back() {
    // MIPS64 (legacy 64-bit struct stat, `_store_mips64` in fstat.py) has
    // no native Rust writer, so the handler intentionally errors out and
    // lets the dispatcher fall back to the Python proc. X86 (angr-11djq.5.1),
    // ARM (angr-11djq.5.2) and MIPS32 (angr-11djq.5.3, via `write_mips32_stat`)
    // ARE supported now, so they are excluded here.
    {
        let arch = "mips64";
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();
        let fd = state
            .file_system()
            .open("/tmp/foo".into(), FdFlags::ReadOnly);

        let err = NativeFstatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(fd as u128, bits),
                    RustBV::concrete(0x4000, bits),
                ],
            )
            .expect_err("{arch}: must surface as Other");
        match err {
            SyscallError::Other(msg) => {
                assert!(msg.contains("unsupported arch"), "{arch}: got {msg:?}",)
            }
            other => panic!("{arch}: expected Other, got {other:?}"),
        }
    }
}

#[test]
fn fstat_symbolic_fd_falls_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    let sym_fd = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "fd", 64)
    };
    let err = NativeFstatSyscall
        .call(&mut state, &[sym_fd, RustBV::concrete(0x4000, 64)])
        .expect_err("must fall back");
    assert_symbolic_arg(err, "fd");
}

#[test]
fn fstat_symbolic_buf_falls_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    let sym_buf = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "statbuf", 64)
    };
    let err = NativeFstatSyscall
        .call(&mut state, &[RustBV::concrete(0, 64), sym_buf])
        .expect_err("must fall back");
    assert_symbolic_arg(err, "statbuf");
}

#[test]
fn fstat_unmapped_buf_surfaces_error() {
    let mut state = RustSimState::new("amd64").expect("state");
    let fd = state.file_system().open("/tmp/x".into(), FdFlags::ReadOnly);
    // Do NOT map the destination page — store should error.
    let err = NativeFstatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 64),
                RustBV::concrete(0x8000, 64),
            ],
        )
        .expect_err("unmapped should error");
    // The MemoryError surfaces as SyscallError via the `?` conversion.
    assert_memory_err(err);
}

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
    let _fd = state.file_system().open_with_content(
        "/tmp/sized".into(),
        FdFlags::ReadOnly,
        b"hello, world!".to_vec(),
    );

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
        .open("/tmp/symstat".into(), FdFlags::ReadOnly);

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

    let _fd = state.file_system().open_with_content(
        "notes.txt".into(),
        FdFlags::ReadOnly,
        b"hello".to_vec(),
    );

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
    let _fd_small = state.file_system().open_with_content(
        "/tmp/shared".into(),
        FdFlags::ReadOnly,
        vec![0u8; 4],
    );
    let _fd_big = state.file_system().open_with_content(
        "/tmp/shared".into(),
        FdFlags::ReadOnly,
        vec![0u8; 17],
    );

    let out = NativeStatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("stat ok");
    assert_eq!(expect_continue(out), 0);
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 17);
}

// ===== lstat (angr-poao) =====
//
// lstat is a stat() clone in our model (no symlinks in FileSystem),
// so most of these mirror the stat tests above. The unsupported-arch
// case differs slightly (lstat dropped on ARM64 — newfstatat is the
// only stat-shaped syscall there).

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

    let _fd = state.file_system().open_with_content(
        "/tmp/lsized".into(),
        FdFlags::ReadOnly,
        b"abc".to_vec(),
    );

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

// ===== newfstatat (angr-poao) =====

/// Same AT_FDCWD constant used by NativeNewfstatatSyscall.
const TEST_AT_FDCWD_NFA: u64 = AT_FDCWD_UNSIGNED;

#[test]
fn newfstatat_unknown_path_returns_minus_one() {
    let mut state = state_with_path(b"/no/such/file");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeNewfstatatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD_NFA as u128, 64),
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
    let _fd = state.file_system().open_with_content(
        "/tmp/nfa.txt".into(),
        FdFlags::ReadOnly,
        b"hello, world!".to_vec(),
    );

    let out = NativeNewfstatatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD_NFA as u128, 64),
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
    let _fd = state.file_system().open_with_content(
        "/tmp/arm-nfa".into(),
        FdFlags::ReadOnly,
        vec![0u8; 4096],
    );

    let out = NativeNewfstatatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD_NFA as u128, 64),
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
                RustBV::concrete(TEST_AT_FDCWD_NFA as u128, 64),
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
                    RustBV::concrete(TEST_AT_FDCWD_NFA as u128, bits),
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
                RustBV::concrete(TEST_AT_FDCWD_NFA as u128, 64),
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
                RustBV::concrete(TEST_AT_FDCWD_NFA as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x8000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect_err("unmapped should error");
    assert_memory_err(err);
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

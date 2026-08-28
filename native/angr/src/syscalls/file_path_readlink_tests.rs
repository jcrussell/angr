//! `readlink` / `readlinkat` tests (angr-wv38).

use super::*;
use super::file_path_tests_support::*;
use crate::memory::Permission;
use crate::state::RustSimState;
use crate::symbolic::RustBV;
use crate::syscalls::{NativeSyscall, SyscallOutcome};

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
    // -1 (EINVAL — not a symlink): the symlink table is empty here.
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

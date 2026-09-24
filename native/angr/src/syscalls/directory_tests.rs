//! Tests for [`super`] — directory / file-name syscall handlers.
//!
//! Extracted from the former inline `#[cfg(test)] mod tests` block in
//! `directory.rs` (angr-7k4r) per the mod-tests sibling-extraction pattern.

use super::*;
use crate::state::RustSimState;
use crate::symbolic::RustBV;
use crate::syscalls::{NativeSyscall, SyscallOutcome};

/// Sweep the 9 stub handlers across every supported arch and verify
/// they return a fresh `RustBV::symbolic` of width `arch().bits()`.
/// Successive invocations must yield distinct fresh symbols.
#[test]
fn dir_stub_handlers_return_fresh_symbolic_on_all_arches() {
    let cases: &[(&'static dyn NativeSyscall, &str, usize)] = &[
        (&NativeFchdirSyscall, "fchdir", 1),
        (&NativeMkdirSyscall, "mkdir", 2),
        (&NativeMkdiratSyscall, "mkdirat", 3),
        (&NativeRmdirSyscall, "rmdir", 1),
        (&NativeUnlinkSyscall, "unlink", 1),
        (&NativeUnlinkatSyscall, "unlinkat", 3),
        (&NativeRenameSyscall, "rename", 2),
        (&NativeRenameatSyscall, "renameat", 4),
        (&NativeRenameat2Syscall, "renameat2", 5),
    ];

    for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();
        for &(handler, label, nargs) in cases {
            assert_eq!(handler.name(), label);
            assert_eq!(handler.num_args(), nargs, "{label} arity");

            let args: Vec<RustBV> = (0..nargs).map(|_| RustBV::concrete(0, bits)).collect();
            let outcome = handler
                .call(&mut state, &args)
                .unwrap_or_else(|e| panic!("{arch} {label}: {e:?}"));
            let ret = match outcome {
                SyscallOutcome::ContinueSymbolic { ret } => ret,
                other => {
                    panic!("{arch} {label}: expected ContinueSymbolic, got {other:?}")
                }
            };
            assert_eq!(ret.width(), bits);
            assert!(ret.as_u64().is_none(), "{arch} {label} should be symbolic");

            let outcome2 = handler.call(&mut state, &args).unwrap();
            let ret2 = match outcome2 {
                SyscallOutcome::ContinueSymbolic { ret } => ret,
                _ => unreachable!(),
            };
            let (id1, id2) = match (&ret, &ret2) {
                (RustBV::Symbolic { id: a, .. }, RustBV::Symbolic { id: b, .. }) => (*a, *b),
                _ => panic!("{arch} {label}: expected Symbolic variant"),
            };
            assert_ne!(
                id1, id2,
                "{arch} {label} successive calls must yield distinct symbol IDs"
            );
        }
    }
}

/// chdir + getcwd round-trip: store path bytes at buf, run chdir,
/// then getcwd into a second buffer and confirm the bytes match.
/// This is the integration acceptance test from the bd description.
#[test]
fn chdir_and_getcwd_round_trip() {
    let mut state = RustSimState::new("amd64").expect("state");

    // Default cwd should be b"/".
    assert_eq!(state.file_system_ref().cwd(), b"/");

    // Map a writable region covering both the chdir path buffer
    // (0x1000) and the getcwd destination buffer (0x2000).
    use crate::memory::Permission;
    state.map_memory_data(0x1000, &[0u8; 0x2000], Permission::RW);

    // Write "/tmp/foo\0" at 0x1000.
    let path = b"/tmp/foo";
    for (i, &b) in path.iter().enumerate() {
        state
            .memory_store(0x1000 + i as u64, RustBV::concrete(b as u128, 8))
            .unwrap();
    }
    state
        .memory_store(0x1000 + path.len() as u64, RustBV::concrete(0, 8))
        .unwrap();

    // chdir(0x1000) — should return 0 and set cwd to "/tmp/foo".
    let outcome = NativeChdirSyscall
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0, "chdir returns 0"),
        other => panic!("chdir: expected Continue, got {other:?}"),
    }
    assert_eq!(state.file_system_ref().cwd(), b"/tmp/foo");

    // getcwd(buf=0x2000, size=64) — should write "/tmp/foo\0" and
    // return 9.
    let outcome = NativeGetcwdSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(64, 64)],
        )
        .unwrap();
    let ret = match outcome {
        SyscallOutcome::Continue { ret } => ret,
        other => panic!("getcwd: expected Continue, got {other:?}"),
    };
    assert_eq!(ret, (path.len() + 1) as u64);

    // Read back the bytes and confirm.
    let mut readback = Vec::new();
    for i in 0..(path.len() + 1) {
        let bv = state.memory_load(0x2000 + i as u64, 1).unwrap();
        readback.push(bv.as_u64().unwrap() as u8);
    }
    let mut expected = path.to_vec();
    expected.push(0);
    assert_eq!(readback, expected, "getcwd writes cwd + NUL");
}

/// getcwd with size < len(cwd)+1 must return -ERANGE without touching
/// the buffer.
#[test]
fn getcwd_returns_erange_when_buf_too_small() {
    let mut state = RustSimState::new("amd64").expect("state");
    // Default cwd "/" + NUL = 2 bytes.
    let outcome = NativeGetcwdSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x3000, 64), RustBV::concrete(1, 64)],
        )
        .unwrap();
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, NEG_ERANGE),
        other => panic!("getcwd undersized: expected Continue, got {other:?}"),
    }
}

/// chdir on a symbolic path falls back to Python via
/// `SyscallError::SymbolicArgument`.
#[test]
fn chdir_symbolic_path_pointer_falls_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    let bits = state.arch().bits();
    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "sym_path_ptr", bits)
    };
    let err = NativeChdirSyscall.call(&mut state, &[sym]).unwrap_err();
    match err {
        SyscallError::SymbolicArgument(_) => (),
        other => panic!("expected SymbolicArgument, got {other:?}"),
    }
}

/// `getcwd` into a buffer that is not mapped at all: the very first
/// `memory_store` faults, so nothing is written and the handler reports
/// `-EFAULT` rather than propagating a `SyscallError`.
#[test]
fn getcwd_unmapped_buf_returns_efault() {
    let mut state = RustSimState::new("amd64").expect("state");
    let outcome = NativeGetcwdSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x5000, 64), RustBV::concrete(64, 64)],
        )
        .unwrap();
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, NEG_EFAULT),
        other => panic!("getcwd unmapped buf: expected Continue, got {other:?}"),
    }
    assert!(
        state.memory_load(0x5000, 1).is_err(),
        "faulting store must not have mapped the page"
    );
}

/// Pins the deliberately non-atomic write loop documented on
/// [`NativeGetcwdSyscall`]: when the payload runs off the end of the
/// mapped region, the bytes that landed before the fault stay committed.
/// A change to all-or-nothing semantics should update that doc and this
/// test together.
#[test]
fn getcwd_partial_write_is_not_rolled_back() {
    use crate::memory::Permission;
    let mut state = RustSimState::new("amd64").expect("state");
    // Exactly one writable page: [0x1000, 0x2000).
    state.map_memory_data(0x1000, &[0u8; 0x1000], Permission::RW);
    state.file_system().set_cwd(b"/abc".to_vec());

    // payload is b"/abc\0" (5 bytes); only 2 of them fit before 0x2000.
    let buf = 0x1000u64 + 0x1000 - 2;
    let outcome = NativeGetcwdSyscall
        .call(
            &mut state,
            &[RustBV::concrete(buf as u128, 64), RustBV::concrete(64, 64)],
        )
        .unwrap();
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, NEG_EFAULT),
        other => panic!("getcwd spanning buf: expected Continue, got {other:?}"),
    }

    for (i, &b) in b"/a".iter().enumerate() {
        let got = state.memory_load(buf + i as u64, 1).unwrap().as_u64();
        assert_eq!(got, Some(b as u64), "byte {i} written before the fault");
    }
}

/// Either `getcwd` argument being symbolic falls back to Python via
/// `SyscallError::SymbolicArgument` — the proc's `solver.eval_one(size)`
/// path handles the unique-but-symbolic case we cannot.
#[test]
fn getcwd_symbolic_args_fall_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    let bits = state.arch().bits();
    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "sym_getcwd_arg", bits)
    };
    for (label, args) in [
        ("buf", [sym.clone(), RustBV::concrete(64, bits)]),
        ("size", [RustBV::concrete(0x1000, bits), sym.clone()]),
    ] {
        let err = NativeGetcwdSyscall.call(&mut state, &args).unwrap_err();
        match err {
            SyscallError::SymbolicArgument(_) => (),
            other => panic!("symbolic {label}: expected SymbolicArgument, got {other:?}"),
        }
    }
}

/// `chdir` on a *concrete* pointer whose path has a symbolic byte before
/// the NUL exercises `read_concrete_cstring`'s mid-string fallback — a
/// different branch from the fully-symbolic pointer covered by
/// `chdir_symbolic_path_pointer_falls_back`.
#[test]
fn chdir_symbolic_byte_mid_path_falls_back() {
    use crate::memory::Permission;
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory_data(0x1000, &[0u8; 0x1000], Permission::RW);

    state
        .memory_store(0x1000, RustBV::concrete(b'/' as u128, 8))
        .unwrap();
    let sym_byte = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "sym_path_byte", 8)
    };
    state.memory_store(0x1001, sym_byte).unwrap();

    let err = NativeChdirSyscall
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap_err();
    match err {
        SyscallError::SymbolicArgument(msg) => {
            assert!(msg.contains("offset 1"), "message names the offset: {msg}");
        }
        other => panic!("expected SymbolicArgument, got {other:?}"),
    }
    // cwd must be untouched by the aborted chdir.
    assert_eq!(state.file_system_ref().cwd(), b"/");
}
